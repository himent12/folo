//! Telling at once that nothing listens on a port of this machine. Windows answers a refused loopback
//! connection only after its SYN retries, about 2 s, so [`super::probe`] of an address where no server runs
//! waited out its whole half-second connect limit on every launch (the default `127.0.0.1:7878` usually has
//! nothing on it: the server the app starts listens on a free port). One connection made with the SYN
//! retries turned off (`SIO_TCP_INITIAL_RTO`) is refused in about a millisecond instead.
//!
//! The check is bounded by one [`DEADLINE`], counted from the call: every address is tried at once, without
//! blocking, and the wait for their answers ends at it. Resolving the name (an address, or `localhost` from
//! the hosts file) counts against it but can't be cut short; once the deadline has passed, no connection is
//! started and none is waited for. A connection still unanswered at the deadline (a filter that drops
//! loopback SYNs silently) is not a refusal, and the probe decides as it did before, [`DEADLINE`] later.
//!
//! A server that listens but whose accept queue is full answers with a reset, which reads as refused here:
//! the app then starts its own `folo serve`, and the data dir's one-owner rule (`serve.lock`) makes the app
//! connect to the server that owns it.

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::{Duration, Instant};

/// How long the whole check may take (the module docs say what it bounds). A refused loopback connection
/// without SYN retries takes about a millisecond; this leaves room for a loaded machine and stays well
/// under the probe's own 0.5 s.
pub const DEADLINE: Duration = Duration::from_millis(100);

/// Whether every address `url` names is on this machine and refused a TCP connection just now: then no
/// server answers there, and the probe can say so without waiting. `false` whenever that is not certain
/// (another machine's address, a name that doesn't resolve, a connection that was accepted, failed some
/// other way or got no answer within [`DEADLINE`], a system without the option, any OS but Windows, where
/// a refusal is immediate anyway): the caller then probes as before. The connection is made directly, so
/// the caller asks only when its HTTP client would connect directly too (no proxy for that address).
pub fn refused(url: &str) -> bool {
    refused_within(url, DEADLINE)
}

/// [`refused`] with its deadline `limit` from now.
fn refused_within(url: &str, limit: Duration) -> bool {
    if !cfg!(windows) {
        return false;
    }
    let deadline = Instant::now() + limit;
    let Ok(uri) = url.parse::<ureq::http::Uri>() else {
        return false;
    };
    let Some(host) = uri.host() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let port = uri.port_u16().unwrap_or(if uri.scheme_str() == Some("https") { 443 } else { 80 });
    // Only addresses (and `localhost`, from the hosts file): a name another resolver must answer is not
    // this machine's, and its lookup would cost what this saves.
    let local_name = host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok();
    if !local_name {
        return false;
    }
    let Ok(addrs) = (host, port).to_socket_addrs() else {
        return false;
    };
    let addrs: Vec<SocketAddr> = addrs.collect();
    !addrs.is_empty() && addrs.iter().all(|a| a.ip().is_loopback()) && all_refused(&addrs, deadline)
}

/// One TCP connection to each of `addrs` at once, without SYN retries: whether every one was refused by
/// `deadline`. A deadline already past, an accepted connection, any other error, a connection still
/// unanswered at the deadline or more addresses than one `select` takes are all `false`.
#[cfg(windows)]
fn all_refused(addrs: &[SocketAddr], deadline: Instant) -> bool {
    use windows_sys::Win32::Networking::WinSock::FD_SETSIZE;
    if addrs.len() > FD_SETSIZE as usize || Instant::now() >= deadline {
        return false;
    }
    // Declared first, so dropped last: every socket is closed before Winsock is released.
    let Some(_wsa) = win::Wsa::start() else { return false };
    let mut pending: Vec<win::Socket> = Vec::with_capacity(addrs.len());
    for addr in addrs {
        match win::start_connect(addr) {
            win::Started::Pending(s) => pending.push(s),
            win::Started::Refused => {}
            win::Started::Other => return false,
        }
    }
    wait_refused(pending, deadline)
}

/// Waits on the connections `pending` (started without blocking, at most `FD_SETSIZE`): `true` once every
/// one was refused, `false` as soon as one connected or failed some other way, or at `deadline` with one
/// still unanswered.
#[cfg(windows)]
fn wait_refused(mut pending: Vec<win::Socket>, deadline: Instant) -> bool {
    use windows_sys::Win32::Networking::WinSock::{
        FD_SET, SO_ERROR, SOCKET_ERROR, SOL_SOCKET, TIMEVAL, WSAECONNREFUSED, getsockopt, select,
    };
    while !pending.is_empty() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        let mut connected = FD_SET::default();
        let mut failed = FD_SET::default();
        for (i, s) in pending.iter().enumerate() {
            connected.fd_array[i] = s.0;
            failed.fd_array[i] = s.0;
        }
        connected.fd_count = pending.len() as u32;
        failed.fd_count = pending.len() as u32;
        // At least a microsecond: a zero `TIMEVAL` would only poll.
        let micros = left.as_micros().max(1);
        let wait = TIMEVAL { tv_sec: (micros / 1_000_000) as i32, tv_usec: (micros % 1_000_000) as i32 };
        // SAFETY: the sets hold only our live sockets (`fd_count` of them) and outlive the call, as does
        // `wait`; the first argument is ignored by Winsock.
        let ready = unsafe { select(0, std::ptr::null_mut(), &mut connected, &mut failed, &wait) };
        if ready < 0 {
            return false;
        }
        // Nothing answered in the time asked: the loop gives up at the deadline (a wait can end a little
        // before it).
        if ready == 0 {
            continue;
        }
        // On return each set holds only its ready sockets. A connection that completed: someone listens.
        if connected.fd_count > 0 {
            return false;
        }
        for &s in &failed.fd_array[..failed.fd_count as usize] {
            let mut error = 0i32;
            let mut len = size_of::<i32>() as i32;
            // SAFETY: `s` is one of our live sockets; `error` and `len` describe a writable i32.
            let got =
                unsafe { getsockopt(s, SOL_SOCKET, SO_ERROR, (&mut error as *mut i32).cast(), &mut len) };
            if got == SOCKET_ERROR || error != WSAECONNREFUSED {
                return false;
            }
            pending.retain(|p| p.0 != s);
        }
    }
    true
}

/// Elsewhere a refused loopback connection fails at once: the probe itself is quick.
#[cfg(not(windows))]
fn all_refused(_addrs: &[SocketAddr], _deadline: Instant) -> bool {
    false
}

/// The Winsock parts of the check, each owning what it opens.
#[cfg(windows)]
mod win {
    use std::net::SocketAddr;

    use windows_sys::Win32::Networking::WinSock::{
        AF_INET, AF_INET6, FIONBIO, INVALID_SOCKET, IPPROTO_TCP, SIO_TCP_INITIAL_RTO, SOCK_STREAM, SOCKADDR,
        SOCKADDR_IN, SOCKADDR_IN6, SOCKET, SOCKET_ERROR, TCP_INITIAL_RTO_PARAMETERS, WSACleanup, WSADATA,
        WSAECONNREFUSED, WSAEWOULDBLOCK, WSAGetLastError, WSAIoctl, WSAStartup, closesocket, connect,
        ioctlsocket, socket,
    };

    /// One `WSAStartup`, released by its `WSACleanup` when dropped (Winsock counts them: std's own start-up
    /// stays valid).
    pub struct Wsa(());

    impl Wsa {
        /// Winsock 2.2.
        pub fn start() -> Option<Wsa> {
            Wsa::start_version(0x0202)
        }

        /// `WSAStartup` of `version`, and its guard only when it succeeded: a start that failed is not
        /// counted, and a `WSACleanup` for it would release another user's start-up.
        pub fn start_version(version: u16) -> Option<Wsa> {
            // SAFETY: `data` is a writable WSADATA.
            let started = unsafe {
                let mut data: WSADATA = std::mem::zeroed();
                WSAStartup(version, &mut data) == 0
            };
            // The guard is made only here (`then_some` would make it first and drop it on failure).
            if started { Some(Wsa(())) } else { None }
        }
    }

    impl Drop for Wsa {
        fn drop(&mut self) {
            // SAFETY: matches the successful `WSAStartup` of `start_version`.
            unsafe { WSACleanup() };
        }
    }

    /// A socket of ours, closed when dropped. Drop it before the [`Wsa`] it was opened under.
    pub struct Socket(pub SOCKET);

    impl Drop for Socket {
        fn drop(&mut self) {
            // SAFETY: the socket is ours and closed only here.
            unsafe { closesocket(self.0) };
        }
    }

    /// `addr` as a Winsock address: the storage (either kind) and its length.
    pub struct Addr {
        v4: SOCKADDR_IN,
        v6: SOCKADDR_IN6,
        pub family: u16,
    }

    impl Addr {
        pub fn new(addr: &SocketAddr) -> Addr {
            let mut a = Addr { v4: SOCKADDR_IN::default(), v6: SOCKADDR_IN6::default(), family: 0 };
            match addr {
                SocketAddr::V4(v4) => {
                    a.family = AF_INET;
                    a.v4.sin_family = AF_INET;
                    a.v4.sin_port = v4.port().to_be();
                    a.v4.sin_addr.S_un.S_addr = u32::from_ne_bytes(v4.ip().octets());
                }
                SocketAddr::V6(v6) => {
                    a.family = AF_INET6;
                    a.v6.sin6_family = AF_INET6;
                    a.v6.sin6_port = v6.port().to_be();
                    a.v6.sin6_flowinfo = v6.flowinfo();
                    a.v6.sin6_addr.u.Byte = v6.ip().octets();
                    a.v6.Anonymous.sin6_scope_id = v6.scope_id();
                }
            }
            a
        }

        /// The pointer and length Winsock takes; valid while `self` is.
        pub fn raw(&self) -> (*const SOCKADDR, i32) {
            if self.family == AF_INET {
                ((&self.v4 as *const SOCKADDR_IN).cast(), size_of::<SOCKADDR_IN>() as i32)
            } else {
                ((&self.v6 as *const SOCKADDR_IN6).cast(), size_of::<SOCKADDR_IN6>() as i32)
            }
        }
    }

    /// A new TCP socket for `addr`'s family.
    pub fn tcp_socket(addr: &Addr) -> Option<Socket> {
        // SAFETY: a plain socket call; the result is checked before it is used.
        let s = unsafe { socket(i32::from(addr.family), SOCK_STREAM, IPPROTO_TCP) };
        // The guard only for a real socket: one made for `INVALID_SOCKET` would close it when dropped.
        if s == INVALID_SOCKET { None } else { Some(Socket(s)) }
    }

    /// How a connection began.
    pub enum Started {
        /// On its way: [`super::all_refused`] waits for it.
        Pending(Socket),
        /// Refused at once.
        Refused,
        /// Anything else (connected at once, failed otherwise, the option not there).
        Other,
    }

    /// Starts a non-blocking TCP connection to `addr` with the SYN retries turned off.
    pub fn start_connect(addr: &SocketAddr) -> Started {
        let addr = Addr::new(addr);
        let Some(s) = tcp_socket(&addr) else { return Started::Other };
        // `TCP_INITIAL_RTO_UNSPECIFIED_RTT` and `TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS` as mstcpip.h defines
        // them, `((USHORT) -1)` and `((UCHAR) -2)` (the fields are a u16 and a u8 in windows-sys too).
        let params = TCP_INITIAL_RTO_PARAMETERS { Rtt: u16::MAX, MaxSynRetransmissions: 0xFE };
        let mut returned = 0u32;
        let mut non_blocking = 1u32;
        let (name, len) = addr.raw();
        // SAFETY: `s` is our live socket; `params`, `returned`, `non_blocking` and `addr` (behind `name`,
        // of length `len`) outlive the calls that read or write them.
        unsafe {
            let set = WSAIoctl(
                s.0,
                SIO_TCP_INITIAL_RTO,
                (&params as *const TCP_INITIAL_RTO_PARAMETERS).cast(),
                size_of::<TCP_INITIAL_RTO_PARAMETERS>() as u32,
                std::ptr::null_mut(),
                0,
                &mut returned,
                std::ptr::null_mut(),
                None,
            );
            if set != 0 || ioctlsocket(s.0, FIONBIO, &mut non_blocking) != 0 {
                return Started::Other;
            }
            if connect(s.0, name, len) != SOCKET_ERROR {
                return Started::Other;
            }
            match WSAGetLastError() {
                WSAEWOULDBLOCK => Started::Pending(s),
                WSAECONNREFUSED => Started::Refused,
                _ => Started::Other,
            }
        }
    }
}

/// For tests here and in [`super`]: a port of ours that nothing listens on.
#[cfg(all(test, windows))]
pub(super) mod closed {
    use std::net::SocketAddr;

    use super::win;

    /// A TCP socket bound to a port but never listening: a connection there is refused, and no other
    /// process can take the port while this lives (a listener bound, read and dropped could be).
    pub struct ClosedPort {
        _socket: win::Socket,
        _wsa: win::Wsa,
        pub port: u16,
    }

    /// A closed port at `addr` (port 0: any free one); `None` when it can't be bound (that address family
    /// missing, or the port taken).
    pub fn closed_port(addr: SocketAddr) -> Option<ClosedPort> {
        use windows_sys::Win32::Networking::WinSock::{SOCKADDR_STORAGE, SOCKET_ERROR, bind, getsockname};
        let wsa = win::Wsa::start()?;
        let a = win::Addr::new(&addr);
        let socket = win::tcp_socket(&a)?;
        let (name, len) = a.raw();
        // SAFETY: our live socket and an address that outlives the call.
        if unsafe { bind(socket.0, name, len) } == SOCKET_ERROR {
            // Both guards hold what they own (a socket, a start-up that succeeded): dropping them is right.
            return None;
        }
        let mut storage = SOCKADDR_STORAGE::default();
        let mut len = size_of::<SOCKADDR_STORAGE>() as i32;
        // SAFETY: our live socket; `storage` is writable for `len` bytes. Both families keep the port in the
        // same place (after the family), in network order.
        let port = unsafe {
            assert_eq!(getsockname(socket.0, (&mut storage as *mut SOCKADDR_STORAGE).cast(), &mut len), 0);
            u16::from_be((&storage as *const SOCKADDR_STORAGE).cast::<u16>().add(1).read_unaligned())
        };
        Some(ClosedPort { _socket: socket, _wsa: wsa, port })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listening_port_is_not_refused() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!refused(&format!("http://127.0.0.1:{port}/api/health")));
    }

    /// The limit of the closed-port checks below: generous for a loaded machine (a refusal without SYN
    /// retries takes about a millisecond), and under the 2 s Windows takes with them, so the option is
    /// still what these tests check.
    #[cfg(windows)]
    const TEST_LIMIT: Duration = Duration::from_secs(1);

    #[cfg(windows)]
    #[test]
    fn a_closed_loopback_port_is_refused_at_once() {
        use closed::closed_port;
        let v4 = closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        assert!(refused_within(&format!("http://127.0.0.1:{}/api/health", v4.port), TEST_LIMIT));

        // IPv6, on a port of its own; a listener of its own there is not refused.
        if let Some(v6) = closed_port("[::1]:0".parse().unwrap()) {
            assert!(refused_within(&format!("http://[::1]:{}", v6.port), TEST_LIMIT));
            let listener = std::net::TcpListener::bind("[::1]:0").expect("IPv6 loopback works: bound above");
            let port = listener.local_addr().unwrap().port();
            assert!(!refused_within(&format!("http://[::1]:{port}"), TEST_LIMIT));
        }

        // `localhost` names both loopback addresses: one port reserved on both, neither listening.
        let both = (0..20).find_map(|_| {
            let v4 = closed_port("127.0.0.1:0".parse().unwrap())?;
            let v6 = closed_port(SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, v4.port)))?;
            Some((v4, v6))
        });
        // None on a machine without IPv6 loopback, where `localhost` is 127.0.0.1 alone (covered above).
        if let Some((v4, _v6)) = both {
            assert!(refused_within(&format!("http://localhost:{}", v4.port), TEST_LIMIT));
        }
    }

    /// One address refused and another still unanswered at the shared deadline: not refused, and the wait
    /// lasts until that deadline (a listening socket stands for the unanswered connection: it is never
    /// writable and never fails, as a connection whose SYN is dropped).
    #[cfg(windows)]
    #[test]
    fn a_connection_unanswered_at_the_deadline_is_not_refused() {
        use std::os::windows::io::IntoRawSocket;
        let closed = closed::closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let _wsa = win::Wsa::start().unwrap();
        let mut pending = Vec::new();
        // Refused at once, or later, in the wait.
        match win::start_connect(&SocketAddr::from(([127, 0, 0, 1], closed.port))) {
            win::Started::Pending(s) => pending.push(s),
            win::Started::Refused => {}
            win::Started::Other => panic!("a closed port neither refused nor pending"),
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        pending.push(win::Socket(listener.into_raw_socket() as usize));
        let limit = Duration::from_millis(50);
        let t = Instant::now();
        assert!(!wait_refused(pending, t + limit));
        let took = t.elapsed();
        assert!(took >= limit, "{took:?}: gave up before the deadline");
        // It ends (with nothing to end it, the wait would last for ever); generous for a loaded machine.
        assert!(took < Duration::from_secs(10), "{took:?}");
    }

    /// Once the deadline has passed, no connection is started: a closed port is not refused then.
    #[cfg(windows)]
    #[test]
    fn nothing_is_tried_after_the_deadline() {
        let closed = closed::closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], closed.port));
        assert!(!all_refused(&[addr], Instant::now()));
        assert!(!refused_within(&format!("http://127.0.0.1:{}", closed.port), Duration::ZERO));
        assert!(all_refused(&[addr], Instant::now() + TEST_LIMIT));
    }

    /// A Winsock start-up that failed is not released: that `WSACleanup` would release another user's.
    /// Run in a process of its own, since std keeps Winsock started in this one once any test used it.
    #[cfg(windows)]
    #[test]
    fn a_failed_winsock_start_releases_nothing() {
        const CHILD: &str = "FOLO_TEST_WSA_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = "api::http::refused::tests::a_failed_winsock_start_releases_nothing";
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--test-threads=1"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let log =
                format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            assert!(out.status.success() && log.contains("1 passed"), "{log}");
            return;
        }
        let ours = win::Wsa::start().expect("Winsock starts");
        // Version 0.0 is below every version Winsock supports: refused.
        assert!(win::Wsa::start_version(0).is_none());
        // Ours still counts: a socket can be made.
        let addr = win::Addr::new(&"127.0.0.1:1".parse().unwrap());
        assert!(win::tcp_socket(&addr).is_some(), "the failed start released ours");
        drop(ours);
    }

    /// A socket that could not be made is not closed: the error left for the caller is the socket call's
    /// own (closing `INVALID_SOCKET` would replace it with WSAENOTSOCK).
    #[cfg(windows)]
    #[test]
    fn a_socket_that_failed_is_not_closed() {
        use windows_sys::Win32::Networking::WinSock::{WSAEAFNOSUPPORT, WSAGetLastError};
        let _wsa = win::Wsa::start().unwrap();
        let mut addr = win::Addr::new(&"127.0.0.1:1".parse().unwrap());
        addr.family = 0x7A7A; // no such address family
        assert!(win::tcp_socket(&addr).is_none());
        // SAFETY: reads this thread's last Winsock error.
        assert_eq!(unsafe { WSAGetLastError() }, WSAEAFNOSUPPORT);
    }

    #[test]
    fn anything_else_is_left_to_the_probe() {
        // Another machine (TEST-NET-1: never connected to), a name, and things that aren't addresses.
        for url in ["http://192.0.2.1:7878", "http://folo.example:7878", "abc def", "", "http://"] {
            assert!(!refused(url), "{url}");
        }
    }
}
