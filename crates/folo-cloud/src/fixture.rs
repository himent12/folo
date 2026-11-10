//! Test support: a scripted HTTP server for the providers' tests (one thread on a loopback port, the answers given
//! in order, an unscripted request answered `599`, every request recorded) and a syntax check of bash scripts.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Command, Stdio};

use folo_test_support::ChildGuard;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// One scripted answer.
#[derive(Clone, Debug)]
pub struct Answer {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Answer {
    /// `status` with a JSON `body`.
    pub fn json(status: u16, body: impl Into<String>) -> Answer {
        Answer {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body.into(),
        }
    }

    /// `status` without a body.
    pub fn empty(status: u16) -> Answer {
        Answer { status, headers: Vec::new(), body: String::new() }
    }

    /// The same with one more header.
    pub fn header(mut self, name: &str, value: &str) -> Answer {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// A request the fixture got.
#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    /// Path and query, as sent.
    pub path: String,
    /// Headers with lowercase names.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// The value of header `name` (lowercase).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    /// The body as JSON.
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

/// The server; it stops when dropped.
pub struct Fixture {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Fixture {
    /// Serves `answers`, one per request, in order.
    pub fn serve(answers: Vec<Answer>) -> Fixture {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (requests, stop) = (requests.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut answers = answers.into_iter();
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut stream) = stream else { continue };
                    let Some(request) = read_request(&stream) else { continue };
                    requests.lock().unwrap().push(request);
                    let answer = answers
                        .next()
                        .unwrap_or_else(|| Answer::json(599, r#"{"detail":"no answer scripted"}"#));
                    write_answer(&mut stream, &answer);
                }
            })
        };
        Fixture { addr, requests, stop, thread: Some(thread) }
    }

    /// `http://127.0.0.1:<port>`.
    pub fn base(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// [`Fixture::base`] plus `path`.
    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base())
    }

    /// The requests so far, in order.
    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }

    /// A loopback URL nothing listens on.
    pub fn closed_url() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("http://{addr}")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the accept loop so it sees the flag.
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_secs(1));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// What `bash -n` (parse only) says of `script`: `Err` with its complaint; `None` when no GNU bash runs here (on
/// Windows `bash` may be WSL's launcher without a distribution).
pub fn bash_syntax(script: &str) -> Option<Result<(), String>> {
    let version = Command::new("bash").arg("--version").output().ok()?;
    if !(version.status.success() && String::from_utf8_lossy(&version.stdout).contains("GNU bash")) {
        return None;
    }
    let child = Command::new("bash")
        .arg("-n")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let mut child = ChildGuard::new(child);
    child.stdin.take().unwrap().write_all(script.as_bytes()).unwrap();
    let mut complaint = String::new();
    child.stderr.as_mut().unwrap().read_to_string(&mut complaint).unwrap();
    Some(if child.wait().unwrap().success() { Ok(()) } else { Err(complaint) })
}

fn read_request(stream: &TcpStream) -> Option<Request> {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next()?.to_string(), parts.next()?.to_string());
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':')?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    let length = headers.iter().find(|(n, _)| n == "content-length").and_then(|(_, v)| v.parse().ok());
    let mut body = vec![0; length.unwrap_or(0)];
    reader.read_exact(&mut body).ok()?;
    Some(Request { method, path, headers, body })
}

fn write_answer(stream: &mut TcpStream, answer: &Answer) {
    let mut head = format!("HTTP/1.1 {} Fixture\r\nConnection: close\r\n", answer.status);
    for (name, value) in &answer.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\n\r\n", answer.body.len()));
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(answer.body.as_bytes());
    let _ = stream.flush();
}
