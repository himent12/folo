//! Telling at once that nothing listens on a port of this machine. Windows answers a refused loopback
//! connection only after its SYN retries, about 2 s, so [`super::probe`] of an address where no server runs
//! waited out its whole half-second connect limit on every launch (the default `127.0.0.1:7878` usually has
//! nothing on it: the server the app starts listens on a free port). One connection made with the SYN
//! retries turned off (`SIO_TCP_INITIAL_RTO`) is refused in about a millisecond instead.
//!
//! The check is bounded by one [`DEADLINE`], counted from the call: every address is tried at once, without
//! blocking, and the wait for their answers ends at it. Resolving the name (an address, or `localhost` from
//! the hosts file) counts against it but can't be cut short. The deadline is looked at right before each
//! connection is started and before a refusal is reported: once it has passed, no connection is started,
//! none is waited for, and a refusal that came after it is not one. A connection still unanswered at the
//! deadline (a filter that drops loopback SYNs silently) is not a refusal, and the probe decides as it did
//! before, [`DEADLINE`] later. The wait is a `select`, which ends on the system timer's ticks: the check can
//! last [`DEADLINE`] plus one tick (about 15 ms on Windows).
//!
//! A server that listens but whose accept queue is full answers with a reset, which reads as refused here:
//! the app then starts its own `folo serve`, and the data dir's one-owner rule (`serve.lock`) makes the app
//! connect to the server that owns it. A queue with one free slot left goes the other way: the check's
//! connection, accepted by the system and closed without a request, holds that slot until the server takes
//! it, so the probe's own connection right after finds the queue full (on Windows it is turned away and
//! times out waiting for its next attempt). So when the check saw a listener ([`Found::Listener`]) and the
//! probe then could not connect, the probe tries once more before it says nobody is there
//! (`server::ask_health`; only a failure to connect, never a connection that was made and then reset).
//!
//! Two corners stay (#308). The second attempt gets its own half-second connect limit, so a server that
//! stops listening between the check and the probe (usually about a millisecond apart, longer on a loaded
//! machine) can take up to about 1.0 s to be found gone instead of 0.5 s, when both attempts wait out their
//! limit (Windows turns a connection away only after its SYN retries; one refused at once ends sooner). And
//! the check's connection is not handed to the probe: on a listener whose accept queue has exactly one
//! free slot and that takes nothing from it for longer than both attempts, both time out connecting and the
//! app says nobody is there. A `folo serve` asks for a queue of 128 (tokio's listener, through mio, on
//! Windows), but it binds its port before it opens its store and only accepts after: during a slow store
//! open, other connections can fill all but one slot. Reusing the check's socket for the health request
//! would close that, but it would need the HTTP client to take a socket it did not open.

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::{Duration, Instant};

/// How long the whole check may take (the module docs say what it bounds). A refused loopback connection
/// without SYN retries takes about a millisecond; this leaves room for a loaded machine and stays well
/// under the probe's own 0.5 s.
pub const DEADLINE: Duration = Duration::from_millis(100);

/// What [`check`] found at an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Found {
    /// Every address refused a connection by the deadline: no server answers there.
    Nothing,
    /// A connection to one of the addresses was accepted: something listens there.
    Listener,
    /// Neither is certain (the cases [`check`] lists).
    Unsure,
}

/// Whether every address `url` names is on this machine and refused a TCP connection just now
/// ([`Found::Nothing`]): then no server answers there, and the probe can say so without waiting.
/// [`Found::Listener`] when a connection was accepted. [`Found::Unsure`] whenever neither is certain
/// (another machine's address, a name that doesn't resolve, a connection that failed some other way or got
/// no answer within [`DEADLINE`], a system without the option, any OS but Windows, where a refusal is
/// immediate anyway): the caller then probes as before. The connection is made directly, so the caller
/// asks only when its HTTP client would connect directly too (no proxy for that address).
pub fn check(url: &str) -> Found {
    check_within(url, DEADLINE)
}

/// [`check`] with its deadline `limit` from now.
fn check_within(url: &str, limit: Duration) -> Found {
    if !cfg!(windows) {
        return Found::Unsure;
    }
    let deadline = Instant::now() + limit;
    let Ok(uri) = url.parse::<ureq::http::Uri>() else {
        return Found::Unsure;
    };
    let Some(host) = uri.host() else {
        return Found::Unsure;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let port = uri.port_u16().unwrap_or(if uri.scheme_str() == Some("https") { 443 } else { 80 });
    // Only addresses (and `localhost`, from the hosts file): a name another resolver must answer is not
    // this machine's, and its lookup would cost what this saves.
    let local_name = host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok();
    if !local_name {
        return Found::Unsure;
    }
    let Ok(addrs) = (host, port).to_socket_addrs() else {
        return Found::Unsure;
    };
    let addrs: Vec<SocketAddr> = addrs.collect();
    if addrs.is_empty() || !addrs.iter().all(|a| a.ip().is_loopback()) {
        return Found::Unsure;
    }
    connect_all(&addrs, deadline, &Instant::now)
}

/// One TCP connection to each of `addrs` at once, without SYN retries: [`Found::Nothing`] when every one
/// was refused by `deadline` (on the clock `now`), [`Found::Listener`] as soon as one was accepted. A
/// deadline already past (also one that passes between two addresses: nothing more is started), any other
/// error, a connection still unanswered at the deadline, a refusal known only after it, or more addresses
/// than one `select` takes are all [`Found::Unsure`].
#[cfg(windows)]
fn connect_all(addrs: &[SocketAddr], deadline: Instant, now: &dyn Fn() -> Instant) -> Found {
    use windows_sys::Win32::Networking::WinSock::FD_SETSIZE;
    if addrs.len() > FD_SETSIZE as usize || now() >= deadline {
        return Found::Unsure;
    }
    // Declared first, so dropped last: every socket is closed before Winsock is released.
    let Some(_wsa) = win::Wsa::start() else { return Found::Unsure };
    let mut pending: Vec<win::Socket> = Vec::with_capacity(addrs.len());
    for addr in addrs {
        match win::start_connect(addr, deadline, now) {
            win::Started::Pending(s) => pending.push(s),
            win::Started::Refused => {}
            win::Started::Connected => return Found::Listener,
            win::Started::Other => return Found::Unsure,
        }
    }
    wait(pending, deadline, now)
}

/// Waits on the connections `pending` (started without blocking, at most `FD_SETSIZE`):
/// [`Found::Nothing`] once every one was refused, if that is before `deadline` (on the clock `now`);
/// [`Found::Listener`] as soon as one connected; [`Found::Unsure`] as soon as one failed some other way, or
/// at `deadline` with one still unanswered.
#[cfg(windows)]
fn wait(mut pending: Vec<win::Socket>, deadline: Instant, now: &dyn Fn() -> Instant) -> Found {
    use windows_sys::Win32::Networking::WinSock::{
        FD_SET, SO_ERROR, SOCKET_ERROR, SOL_SOCKET, TIMEVAL, WSAECONNREFUSED, getsockopt, select,
    };
    while !pending.is_empty() {
        let left = deadline.saturating_duration_since(now());
        if left.is_zero() {
            return Found::Unsure;
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
            return Found::Unsure;
        }
        // Nothing answered in the time asked: the loop gives up at the deadline (a wait can end a little
        // before it).
        if ready == 0 {
            continue;
        }
        // On return each set holds only its ready sockets. A connection that completed: someone listens.
        if connected.fd_count > 0 {
            return Found::Listener;
        }
        for &s in &failed.fd_array[..failed.fd_count as usize] {
            let mut error = 0i32;
            let mut len = size_of::<i32>() as i32;
            // SAFETY: `s` is one of our live sockets; `error` and `len` describe a writable i32.
            let got =
                unsafe { getsockopt(s, SOL_SOCKET, SO_ERROR, (&mut error as *mut i32).cast(), &mut len) };
            if got == SOCKET_ERROR || error != WSAECONNREFUSED {
                return Found::Unsure;
            }
            pending.retain(|p| p.0 != s);
        }
    }
    // Every one refused, but a refusal known only after the deadline is not reported as one.
    if now() >= deadline { Found::Unsure } else { Found::Nothing }
}

/// Elsewhere a refused loopback connection fails at once: the probe itself is quick.
#[cfg(not(windows))]
fn connect_all(_addrs: &[SocketAddr], _deadline: Instant, _now: &dyn Fn() -> Instant) -> Found {
    Found::Unsure
}

/// The Winsock parts of the check, each owning what it opens.
#[cfg(windows)]
mod win {
    use std::net::SocketAddr;
    use std::time::Instant;

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

    impl Socket {
        /// The socket, no longer closed by this guard (its new owner closes it).
        #[cfg(test)]
        pub fn into_raw(self) -> SOCKET {
            let s = self.0;
            std::mem::forget(self);
            s
        }
    }

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
        /// On its way: [`super::wait`] waits for it.
        Pending(Socket),
        /// Refused at once.
        Refused,
        /// Connected at once: something listens.
        Connected,
        /// Anything else (failed otherwise, the option not there, the deadline passed).
        Other,
    }

    #[cfg(test)]
    thread_local! {
        /// Connections this thread started (`connect` called), for the deadline tests.
        static CONNECTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// The connections this thread started so far. For tests.
    #[cfg(test)]
    pub fn connects() -> usize {
        CONNECTS.with(std::cell::Cell::get)
    }

    /// Starts a non-blocking TCP connection to `addr` with the SYN retries turned off, unless `deadline`
    /// (on the clock `now`) has passed by the time it would start.
    pub fn start_connect(addr: &SocketAddr, deadline: Instant, now: &dyn Fn() -> Instant) -> Started {
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
            // Looked at right before the connection starts: the set-up, or an earlier address, may have
            // taken this thread past it.
            if now() >= deadline {
                return Started::Other;
            }
            #[cfg(test)]
            CONNECTS.with(|c| c.set(c.get() + 1));
            if connect(s.0, name, len) != SOCKET_ERROR {
                return Started::Connected;
            }
            match WSAGetLastError() {
                WSAEWOULDBLOCK => Started::Pending(s),
                WSAECONNREFUSED => Started::Refused,
                _ => Started::Other,
            }
        }
    }
}

/// For tests here and in [`super`]: a port of ours that nothing listens on, and a listener with a short
/// accept queue.
#[cfg(all(test, windows))]
pub(super) mod ports {
    use std::net::SocketAddr;

    use super::win;

    /// A listener on a free loopback port whose accept queue holds `backlog` connections (std's listeners
    /// take a long one).
    pub fn listener(backlog: i32) -> std::net::TcpListener {
        use std::os::windows::io::FromRawSocket;
        use windows_sys::Win32::Networking::WinSock::{bind, listen};
        let wsa = win::Wsa::start().expect("Winsock starts");
        let a = win::Addr::new(&"127.0.0.1:0".parse().unwrap());
        let socket = win::tcp_socket(&a).expect("a socket");
        let (name, len) = a.raw();
        // SAFETY: our live socket and an address that outlives the calls.
        unsafe {
            assert_eq!(bind(socket.0, name, len), 0);
            assert_eq!(listen(socket.0, backlog), 0);
        }
        // Winsock stays started for the listener's life (std's listener does not start it); a test process.
        std::mem::forget(wsa);
        // SAFETY: a live listening socket, handed to the listener, which closes it.
        unsafe { std::net::TcpListener::from_raw_socket(socket.into_raw() as u64) }
    }

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
        let found = check(&format!("http://127.0.0.1:{port}/api/health"));
        // Seen as a listener where the check runs (Windows); unsure elsewhere.
        let expected = if cfg!(windows) { Found::Listener } else { Found::Unsure };
        assert_eq!(found, expected);
    }

    /// The limit of the closed-port checks below: generous for a loaded machine (a refusal without SYN
    /// retries takes about a millisecond), and under the 2 s Windows takes with them, so the option is
    /// still what these tests check.
    #[cfg(windows)]
    const TEST_LIMIT: Duration = Duration::from_secs(1);

    #[cfg(windows)]
    #[test]
    fn a_closed_loopback_port_is_refused_at_once() {
        use ports::closed_port;
        let refused = |url: String| check_within(&url, TEST_LIMIT) == Found::Nothing;
        let v4 = closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        assert!(refused(format!("http://127.0.0.1:{}/api/health", v4.port)));

        // IPv6, on a port of its own; a listener of its own there is not refused.
        if let Some(v6) = closed_port("[::1]:0".parse().unwrap()) {
            assert!(refused(format!("http://[::1]:{}", v6.port)));
            let listener = std::net::TcpListener::bind("[::1]:0").expect("IPv6 loopback works: bound above");
            let port = listener.local_addr().unwrap().port();
            assert_eq!(check_within(&format!("http://[::1]:{port}"), TEST_LIMIT), Found::Listener);
        }

        // `localhost` names both loopback addresses: one port reserved on both, neither listening.
        let both = (0..20).find_map(|_| {
            let v4 = closed_port("127.0.0.1:0".parse().unwrap())?;
            let v6 = closed_port(SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, v4.port)))?;
            Some((v4, v6))
        });
        // None on a machine without IPv6 loopback, where `localhost` is 127.0.0.1 alone (covered above).
        if let Some((v4, _v6)) = both {
            assert!(refused(format!("http://localhost:{}", v4.port)));
        }
    }

    /// One address refused and another still unanswered at the shared deadline: not refused, and the wait
    /// lasts until that deadline (a listening socket stands for the unanswered connection: it is never
    /// writable and never fails, as a connection whose SYN is dropped).
    #[cfg(windows)]
    #[test]
    fn a_connection_unanswered_at_the_deadline_is_not_refused() {
        use std::os::windows::io::IntoRawSocket;
        let closed = ports::closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let _wsa = win::Wsa::start().unwrap();
        let mut pending = Vec::new();
        // Refused at once, or later, in the wait.
        let addr = SocketAddr::from(([127, 0, 0, 1], closed.port));
        match win::start_connect(&addr, Instant::now() + TEST_LIMIT, &Instant::now) {
            win::Started::Pending(s) => pending.push(s),
            win::Started::Refused => {}
            _ => panic!("a closed port neither refused nor pending"),
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        pending.push(win::Socket(listener.into_raw_socket() as usize));
        let limit = Duration::from_millis(50);
        let t = Instant::now();
        assert_eq!(wait(pending, t + limit, &Instant::now), Found::Unsure);
        let took = t.elapsed();
        assert!(took >= limit, "{took:?}: gave up before the deadline");
        // It ends (with nothing to end it, the wait would last for ever); generous for a loaded machine.
        assert!(took < Duration::from_secs(10), "{took:?}");
    }

    /// Once the deadline has passed, no connection is started and no refusal is reported: a closed port
    /// is not refused then. Also when the deadline passes between two addresses (a clock that jumps past it
    /// once the first connection started): the second is not started.
    #[cfg(windows)]
    #[test]
    fn nothing_is_tried_after_the_deadline() {
        let closed = ports::closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let other = ports::closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], closed.port));
        let addr2 = SocketAddr::from(([127, 0, 0, 1], other.port));
        assert_eq!(connect_all(&[addr], Instant::now(), &Instant::now), Found::Unsure);
        assert_eq!(check_within(&format!("http://127.0.0.1:{}", closed.port), Duration::ZERO), Found::Unsure);
        assert_eq!(connect_all(&[addr, addr2], Instant::now() + TEST_LIMIT, &Instant::now), Found::Nothing);

        let start = Instant::now();
        let deadline = start + TEST_LIMIT;
        let before = win::connects();
        let clock = || if win::connects() > before { deadline } else { start };
        assert_eq!(connect_all(&[addr, addr2], deadline, &clock), Found::Unsure);
        assert_eq!(win::connects() - before, 1, "a connection started after the deadline");

        // Every connection refused, but only known after the deadline: not a refusal.
        let _wsa = win::Wsa::start().unwrap();
        assert_eq!(wait(Vec::new(), deadline, &|| deadline), Found::Unsure);
        assert_eq!(wait(Vec::new(), deadline, &|| start), Found::Nothing);
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
            assert_eq!(check(url), Found::Unsure, "{url}");
        }
    }
}
