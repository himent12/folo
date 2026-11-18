//! Telling at once that nothing listens on a port of this machine. Windows answers a refused loopback
//! connection only after its SYN retries, about 2 s, so [`super::probe`] of an address where no server runs
//! waited out its whole half-second connect limit on every launch (the default `127.0.0.1:7878` usually has
//! nothing on it: the server the app starts listens on a free port). One connection made with the SYN
//! retries turned off (`SIO_TCP_INITIAL_RTO`) is refused in about a millisecond instead.

use std::net::{SocketAddr, ToSocketAddrs};

/// Whether every address `url` names is on this machine and refused a TCP connection just now: then no
/// server answers there, and the probe can say so without waiting. `false` whenever that is not certain
/// (another machine's address, a name that doesn't resolve, a connection that was accepted or failed
/// some other way, a system without the option, any OS but Windows, where a refusal is immediate anyway):
/// the caller then probes as before.
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
    !addrs.is_empty() && addrs.iter().all(|a| a.ip().is_loopback() && connect_refused(a))
}

/// One TCP connection to `addr` without SYN retries: whether it was refused.
#[cfg(windows)]
fn connect_refused(addr: &SocketAddr) -> bool {
    use windows_sys::Win32::Networking::WinSock::{
        AF_INET, AF_INET6, INVALID_SOCKET, IPPROTO_TCP, SIO_TCP_INITIAL_RTO, SOCK_STREAM, SOCKADDR,
        SOCKADDR_IN, SOCKADDR_IN6, SOCKET_ERROR, TCP_INITIAL_RTO_PARAMETERS, WSADATA, WSAECONNREFUSED,
        WSAGetLastError, WSAIoctl, WSAStartup, closesocket, connect, socket,
    };
    // `TCP_INITIAL_RTO_UNSPECIFIED_RTT` and `TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS` as mstcpip.h defines
    // them, `((USHORT) -1)` and `((UCHAR) -2)` (windows-sys widens the second to a u16).
    let params = TCP_INITIAL_RTO_PARAMETERS { Rtt: u16::MAX, MaxSynRetransmissions: 0xFE };
    let mut v4 = SOCKADDR_IN::default();
    let mut v6 = SOCKADDR_IN6::default();
    let (family, name, len) = match addr {
        SocketAddr::V4(a) => {
            v4.sin_family = AF_INET;
            v4.sin_port = a.port().to_be();
            v4.sin_addr.S_un.S_addr = u32::from_ne_bytes(a.ip().octets());
            (AF_INET, &v4 as *const SOCKADDR_IN as *const SOCKADDR, size_of::<SOCKADDR_IN>())
        }
        SocketAddr::V6(a) => {
            v6.sin6_family = AF_INET6;
            v6.sin6_port = a.port().to_be();
            v6.sin6_flowinfo = a.flowinfo();
            v6.sin6_addr.u.Byte = a.ip().octets();
            v6.Anonymous.sin6_scope_id = a.scope_id();
            (AF_INET6, &v6 as *const SOCKADDR_IN6 as *const SOCKADDR, size_of::<SOCKADDR_IN6>())
        }
    };
    // SAFETY: plain Winsock calls. `WSAStartup` is reference counted (std's own start-up of Winsock stays
    // valid; this one is never matched by `WSACleanup`, which std doesn't do either). The socket is ours
    // and closed on every path below; `params`, `name` (pointing at `v4` or `v6`, live locals of the
    // length given) and `returned` outlive the calls that read or write them.
    unsafe {
        let mut data: WSADATA = std::mem::zeroed();
        if WSAStartup(0x0202, &mut data) != 0 {
            return false;
        }
        let s = socket(i32::from(family), SOCK_STREAM, IPPROTO_TCP);
        if s == INVALID_SOCKET {
            return false;
        }
        let mut returned = 0u32;
        let set = WSAIoctl(
            s,
            SIO_TCP_INITIAL_RTO,
            &params as *const TCP_INITIAL_RTO_PARAMETERS as *const core::ffi::c_void,
            size_of::<TCP_INITIAL_RTO_PARAMETERS>() as u32,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null_mut(),
            None,
        );
        let refused =
            set == 0 && connect(s, name, len as i32) == SOCKET_ERROR && WSAGetLastError() == WSAECONNREFUSED;
        closesocket(s);
        refused
    }
}

/// Elsewhere a refused loopback connection fails at once: the probe itself is quick.
#[cfg(not(windows))]
fn connect_refused(_addr: &SocketAddr) -> bool {
    false
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

    #[cfg(windows)]
    #[test]
    fn a_closed_loopback_port_is_refused_at_once() {
        use std::time::{Duration, Instant};
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let t = Instant::now();
        assert!(refused(&format!("http://127.0.0.1:{port}/api/health")));
        // Without the option Windows takes about 2 s (and the probe gave up after 0.5 s); with it, about
        // a millisecond. The limit leaves room for a loaded machine.
        assert!(t.elapsed() < Duration::from_millis(400), "{:?}", t.elapsed());
        // `localhost` names both loopback addresses; neither listens.
        assert!(refused(&format!("http://localhost:{port}")));
        let v6 = std::net::TcpListener::bind("[::1]:0");
        if let Ok(v6) = v6 {
            let port = v6.local_addr().unwrap().port();
            assert!(!refused(&format!("http://[::1]:{port}")));
            drop(v6);
            assert!(refused(&format!("http://[::1]:{port}")));
        }
    }

    #[test]
    fn anything_else_is_left_to_the_probe() {
        // Another machine (TEST-NET-1: never connected to), a name, and things that aren't addresses.
        for url in ["http://192.0.2.1:7878", "http://folo.example:7878", "abc def", "", "http://"] {
            assert!(!refused(url), "{url}");
        }
    }
}
