//! Telling at once that nothing listens on a port of this machine. Windows answers a refused loopback
//! connection only after its SYN retries, about 2 s, so [`super::probe`] of an address where no server runs
//! waited out its whole half-second connect limit on every launch (the default `127.0.0.1:7878` usually has
//! nothing on it: the server the app starts listens on a free port). One connection made with the SYN
//! retries turned off (`SIO_TCP_INITIAL_RTO`) is refused in about a millisecond instead.
//!
//! The check is bounded: every address is tried at once, without blocking, and all of them share one
//! [`DEADLINE`]. A connection still unanswered then (a filter that drops loopback SYNs silently) is not a
//! refusal, and the probe decides as it did before, [`DEADLINE`] later.

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::Duration;

/// How long the whole check may take. A refused loopback connection without SYN retries takes about a
/// millisecond; this leaves room for a loaded machine and stays well under the probe's own 0.5 s.
pub const DEADLINE: Duration = Duration::from_millis(100);

/// Whether every address `url` names is on this machine and refused a TCP connection just now: then no
/// server answers there, and the probe can say so without waiting. `false` whenever that is not certain
/// (another machine's address, a name that doesn't resolve, a connection that was accepted, failed some
/// other way or got no answer within [`DEADLINE`], a system without the option, any OS but Windows, where
/// a refusal is immediate anyway): the caller then probes as before. The connection is made directly, so
/// the caller asks only when its HTTP client would connect directly too (no proxy for that address).
pub fn refused(url: &str) -> bool {
    if !cfg!(windows) {
        return false;
    }
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
    !addrs.is_empty() && addrs.iter().all(|a| a.ip().is_loopback()) && all_refused(&addrs, DEADLINE)
}

/// One TCP connection to each of `addrs` at once, without SYN retries: whether every one was refused
/// within `limit`. An accepted connection, any other error, a timeout or more addresses than one `select`
/// takes are all `false`.
#[cfg(windows)]
fn all_refused(addrs: &[SocketAddr], limit: Duration) -> bool {
    use windows_sys::Win32::Networking::WinSock::{
        FD_SET, FD_SETSIZE, SO_ERROR, SOCKET_ERROR, SOL_SOCKET, TIMEVAL, WSAECONNREFUSED, getsockopt, select,
    };
    let deadline = std::time::Instant::now() + limit;
    if addrs.len() > FD_SETSIZE as usize {
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
    while !pending.is_empty() {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
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
        if ready <= 0 {
            // An error, or nothing answered in time: not known to be refused.
            return false;
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
fn all_refused(_addrs: &[SocketAddr], _limit: Duration) -> bool {
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
        pub fn start() -> Option<Wsa> {
            // SAFETY: `data` is a writable WSADATA; a start that failed is not counted and not released.
            let started = unsafe {
                let mut data: WSADATA = std::mem::zeroed();
                WSAStartup(0x0202, &mut data) == 0
            };
            started.then_some(Wsa(()))
        }
    }

    impl Drop for Wsa {
        fn drop(&mut self) {
            // SAFETY: matches the successful `WSAStartup` of `start`.
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
        (s != INVALID_SOCKET).then_some(Socket(s))
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
        // them, `((USHORT) -1)` and `((UCHAR) -2)` (windows-sys widens the second to a u16).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listening_port_is_not_refused() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!refused(&format!("http://127.0.0.1:{port}/api/health")));
    }

    /// A port of ours that nothing listens on: a TCP socket bound to it but never listening, so no other
    /// process can take the port while the test runs (a listener bound, read and dropped could be).
    #[cfg(windows)]
    struct Reserved {
        socket: win::Socket,
        _wsa: win::Wsa,
    }

    #[cfg(windows)]
    fn reserve(addr: SocketAddr) -> Option<Reserved> {
        use windows_sys::Win32::Networking::WinSock::{SOCKET_ERROR, bind};
        let wsa = win::Wsa::start()?;
        let a = win::Addr::new(&addr);
        let socket = win::tcp_socket(&a)?;
        let (name, len) = a.raw();
        // SAFETY: our live socket and an address that outlives the call.
        (unsafe { bind(socket.0, name, len) } != SOCKET_ERROR).then_some(Reserved { socket, _wsa: wsa })
    }

    #[cfg(windows)]
    fn port_of(r: &Reserved) -> u16 {
        use windows_sys::Win32::Networking::WinSock::{SOCKADDR_STORAGE, getsockname};
        let mut storage = SOCKADDR_STORAGE::default();
        let mut len = size_of::<SOCKADDR_STORAGE>() as i32;
        // SAFETY: our live socket; `storage` is writable for `len` bytes. Both families keep the port in the
        // same place (after the family), in network order.
        unsafe {
            assert_eq!(getsockname(r.socket.0, (&mut storage as *mut SOCKADDR_STORAGE).cast(), &mut len), 0);
            let raw = (&storage as *const SOCKADDR_STORAGE).cast::<u16>().add(1).read_unaligned();
            u16::from_be(raw)
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_closed_loopback_port_is_refused_at_once() {
        use std::time::Instant;
        let v4 = reserve("127.0.0.1:0".parse().unwrap()).unwrap();
        let port = port_of(&v4);
        let t = Instant::now();
        assert!(refused(&format!("http://127.0.0.1:{port}/api/health")));
        // Without the option Windows takes about 2 s (and the probe gave up after 0.5 s); with it, about
        // a millisecond, and never longer than the deadline.
        assert!(t.elapsed() < DEADLINE * 2, "{:?}", t.elapsed());

        // IPv6, on a port of its own.
        if let Some(v6) = reserve("[::1]:0".parse().unwrap()) {
            let port = port_of(&v6);
            assert!(refused(&format!("http://[::1]:{port}")));
            // Listening there is not refused.
            drop(v6);
            if let Ok(listener) = std::net::TcpListener::bind(("::1", port)) {
                assert!(!refused(&format!("http://[::1]:{port}")));
                drop(listener);
            }
        }

        // `localhost` names both loopback addresses: one port reserved on both, neither listening.
        let both = (0..20).find_map(|_| {
            let v4 = reserve("127.0.0.1:0".parse().unwrap())?;
            let port = port_of(&v4);
            let v6 = reserve(SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port)))?;
            Some((port, v4, v6))
        });
        // None on a machine without IPv6 loopback, where `localhost` is 127.0.0.1 alone (covered above).
        if let Some((port, _v4, _v6)) = both {
            assert!(refused(&format!("http://localhost:{port}")));
        }
    }

    /// A connection nobody answers (TEST-NET-1 is never routed to a host: the SYN is dropped or told
    /// unreachable) gives up within the limit instead of waiting out Windows' connect timeout of seconds,
    /// and is not a refusal.
    #[test]
    fn an_unanswered_connection_is_not_refused_and_ends_at_the_limit() {
        use std::time::Instant;
        let t = Instant::now();
        assert!(!all_refused(&["192.0.2.1:7878".parse().unwrap()], Duration::from_millis(50)));
        assert!(t.elapsed() < Duration::from_millis(400), "{:?}", t.elapsed());
    }

    #[test]
    fn anything_else_is_left_to_the_probe() {
        // Another machine (TEST-NET-1: never connected to), a name, and things that aren't addresses.
        for url in ["http://192.0.2.1:7878", "http://folo.example:7878", "abc def", "", "http://"] {
            assert!(!refused(url), "{url}");
        }
    }
}
