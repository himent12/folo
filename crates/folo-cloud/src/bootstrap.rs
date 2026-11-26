//! The script a provisioned instance runs to become a worker (docs/CLOUD.md, `runpod` → Worker).
//!
//! A provider starts a stock Linux image (Ubuntu 22.04 with bash and apt) with [`script`] as its command and the
//! worker's token in the environment variable `FOLO_TOKEN`. The script, in order:
//! 1. arms the deadline watchdog: after the run's lifetime plus [`GRACE_S`] the instance terminates itself, whatever
//!    the rest of the script is doing (nothing later can disarm it);
//! 2. installs what the image lacks (`ca-certificates`, `curl`, the Vulkan loader);
//! 3. makes sure the Vulkan loader has a manifest for the NVIDIA driver (FOLO's GPU backend is wgpu on Vulkan);
//! 4. installs `folo` with the release's `install.sh`, which verifies the archive's SHA-256;
//! 5. checks that `folo` finds the GPU, and when it does not, points the Vulkan loader at the driver's EGL library
//!    instead of its GLX one. The manifest the NVIDIA runtime mounts names the GLX library, which gives Vulkan no
//!    device in a container without an X server (seen on RunPod, driver 595: the loader cannot get
//!    `vkCreateInstance` from it, with or without the X client libraries installed); the EGL library is the same
//!    driver's headless entry;
//! 6. prints `folo doctor` (the adapters, in the instance's log) and runs `folo serve` in the foreground.
//!
//! However the script ends (a failed step, `folo serve` exiting or crashing), the instance terminates itself after
//! a short pause, so a broken worker does not idle at GPU prices. The token is never part of the script.

use crate::{Error, ErrorKind, Result};

/// Seconds an instance outlives its run's lifetime before it terminates itself: the control plane stops the run
/// at the deadline and collects its checkpoint in that time.
pub const GRACE_S: u64 = 600;

/// Seconds the script keeps trying to terminate the instance once it is time (5 tries, 30 s apart). After that it
/// gives up: an instance that cannot reach its provider's API stays until something else terminates it.
pub const TERMINATE_TRIES_S: u64 = 150;

/// Seconds the script waits after it ended before the instance terminates itself, so the end of its log can still
/// be read in the provider's console.
const EXIT_PAUSE_S: u32 = 60;

/// What a worker's script is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec<'a> {
    /// Where `install.sh` and the release archive are downloaded from ([`crate::worker_base_url`]): an `http://`
    /// or `https://` URL without a trailing slash.
    pub base_url: &'a str,
    /// The port `folo serve` listens on ([`crate::WORKER_PORT`]).
    pub port: u16,
    /// The instance terminates itself this many seconds (plus [`GRACE_S`]) after the script starts.
    pub lifetime_s: u64,
    /// The provider's shell commands that terminate this instance from inside it: the body of a shell function
    /// that succeeds once the provider accepted the termination. Written by the provider, not interpolated from
    /// user input.
    pub terminate: &'a str,
}

/// Checks that `url` can be where a worker downloads FOLO from: `http://` or `https://`, a host (no embedded
/// credentials, query, or fragment), and no spaces or control characters ([`ErrorKind::Invalid`] otherwise,
/// naming the [`crate::WORKER_BASE_URL_FIELD`] field). Quotes and `$` in the path are fine: [`script`] quotes
/// the URL.
pub fn check_base_url(url: &str) -> Result<()> {
    let invalid =
        |why: &str| Error::new(ErrorKind::Invalid, format!("`{}` {why}", crate::WORKER_BASE_URL_FIELD));
    if !url.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(invalid("must not contain spaces or control characters"));
    }
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| invalid("must start with https:// or http://"))?;
    if rest.contains('?') || rest.contains('#') {
        return Err(invalid("must not have a query or a fragment"));
    }
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.contains('@') {
        return Err(invalid("must not contain a user name or password"));
    }
    let (host, port) = if let Some(after_open) = authority.strip_prefix('[') {
        let (h, tail) = after_open.split_once(']').ok_or_else(|| invalid("has an invalid IPv6 host"))?;
        if h.parse::<std::net::Ipv6Addr>().is_err() {
            return Err(invalid("has an invalid IPv6 host"));
        }
        let port = match tail {
            "" => None,
            s => Some(s.strip_prefix(':').ok_or_else(|| invalid("has an invalid host or port"))?),
        };
        (h, port)
    } else if let Some((h, p)) = authority.split_once(':') {
        (h, Some(p))
    } else {
        (authority, None)
    };
    if host.is_empty() || host.contains(['[', ']']) {
        return Err(invalid("has no host"));
    }
    if let Some(p) = port {
        let valid_port = (!p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse::<u16>().ok())
            .flatten()
            .is_some_and(|n| n > 0);
        if !valid_port {
            return Err(invalid("has an invalid port"));
        }
    }
    Ok(())
}

/// The bash script for `spec`; [`ErrorKind::Invalid`] when its URL is not one ([`check_base_url`]).
pub fn script(spec: &Spec) -> Result<String> {
    check_base_url(spec.base_url)?;
    let base = quote(spec.base_url.trim_end_matches('/'));
    let Spec { port, lifetime_s, terminate, .. } = *spec;
    let watchdog_s = lifetime_s.saturating_add(GRACE_S);
    Ok(format!(
        r#"#!/bin/bash
# A FOLO cloud worker (generated by folo-cloud; docs/CLOUD.md). The token comes from $FOLO_TOKEN.
set -euo pipefail

BASE={base}
PORT={port}

log() {{ printf '[folo-worker] %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }}

# The provider's way to terminate this instance from inside it.
self_terminate() {{
{terminate}
}}

# Terminates the instance, retried: the provider's API may be briefly unavailable.
terminate_instance() {{
  local attempt
  for attempt in 1 2 3 4 5; do
    log "terminating this instance (attempt $attempt)"
    if self_terminate; then
      return 0
    fi
    sleep 30
  done
  log "error: could not terminate this instance; the control plane terminates it"
  return 1
}}

# 1. The deadline watchdog, armed before anything that can fail: whatever happens below, this instance does not
#    outlive the run's lifetime plus the grace period.
log "deadline watchdog: terminating in {watchdog_s} s"
( sleep {watchdog_s}; log "deadline reached"; terminate_instance ) &

# Every way out of this script (a failed step, folo serve exiting or crashing) ends the instance, so a broken
# worker does not idle at GPU prices.
on_exit() {{
  local status=$?
  set +e
  log "the worker stopped (exit status $status); terminating this instance in {EXIT_PAUSE_S} s"
  sleep {EXIT_PAUSE_S}
  terminate_instance
}}
trap on_exit EXIT

# 2. What a stock image lacks.
log "installing ca-certificates, curl and the Vulkan loader"
export DEBIAN_FRONTEND=noninteractive
export HOME="${{HOME:-/root}}"
apt_install() {{
  apt-get update -q && apt-get install -y -q --no-install-recommends ca-certificates curl libvulkan1
}}
if ! apt_install; then
  log "apt-get failed; trying again in 15 s"
  sleep 15
  apt_install
fi

# 3. The NVIDIA container runtime mounts the driver's Vulkan library (libGLX_nvidia.so.0) when
#    NVIDIA_DRIVER_CAPABILITIES includes `graphics`, but plain images often lack the ICD manifest that points the
#    Vulkan loader at it, and without one wgpu finds no GPU. Write the manifest the driver package would install.
log "checking the NVIDIA Vulkan driver"
if [ -e /etc/vulkan/icd.d/nvidia_icd.json ] || [ -e /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  log "the Vulkan ICD manifest is present"
elif ldconfig -p | grep 'libGLX_nvidia\.so\.0' >/dev/null; then
  mkdir -p /etc/vulkan/icd.d
  cat > /etc/vulkan/icd.d/nvidia_icd.json <<'EOF_ICD'
{{ "file_format_version": "1.0.0", "ICD": {{ "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0" }} }}
EOF_ICD
  log "wrote /etc/vulkan/icd.d/nvidia_icd.json"
else
  log "warning: no NVIDIA Vulkan driver (libGLX_nvidia.so.0) is mounted; folo will find no GPU"
fi

# 4. folo, with the release's own installer (it verifies the archive's SHA-256).
log "installing folo from $BASE"
cd "$(mktemp -d)"
curl -fsSL --retry 3 --proto '=https,http' -o install.sh "$BASE/install.sh"
sh install.sh --base-url "$BASE" --dir /usr/local/bin

# 5. The GPU, as folo sees it (0x10de = 4318 is NVIDIA). The manifest's GLX library gives Vulkan no device in a
#    container without an X server; the driver's EGL library is its headless entry, so when folo finds no GPU the
#    loader is pointed at a manifest for that one instead.
has_gpu() {{
  local doctor
  doctor="$(folo doctor --json 2>/dev/null || true)"
  case "$doctor" in *'"vendor":4318'*) return 0 ;; *) return 1 ;; esac
}}
log "looking for the GPU"
if has_gpu; then
  log "folo finds the NVIDIA GPU"
else
  mkdir -p /usr/local/share/folo
  cat > /usr/local/share/folo/nvidia_egl_icd.json <<'EOF_EGL'
{{ "file_format_version": "1.0.0", "ICD": {{ "library_path": "libEGL_nvidia.so.0", "api_version": "1.3.0" }} }}
EOF_EGL
  # Loaders before 1.3.207 read VK_ICD_FILENAMES, later ones VK_DRIVER_FILES.
  export VK_DRIVER_FILES=/usr/local/share/folo/nvidia_egl_icd.json
  export VK_ICD_FILENAMES=/usr/local/share/folo/nvidia_egl_icd.json
  if has_gpu; then
    log "folo finds the NVIDIA GPU through the driver's EGL library"
  else
    unset VK_DRIVER_FILES VK_ICD_FILENAMES
    log "warning: folo finds no NVIDIA GPU; the control plane will refuse this worker"
  fi
fi

# 6. The worker.
log "folo doctor"
folo doctor --tensor-cores || log "warning: folo doctor failed"
if [ -z "${{FOLO_TOKEN:-}}" ]; then
  log "error: FOLO_TOKEN is not set; refusing to serve without a token"
  exit 1
fi
log "starting folo serve on port $PORT"
folo serve --host 0.0.0.0 --port "$PORT" --tensor-cores
"#
    ))
}

/// `s` as one bash word that expands to exactly `s`.
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::bash_syntax;

    const TERMINATE: &str = r#"  curl -fsS -X DELETE "https://provider.example/instances/$INSTANCE_ID""#;

    fn spec(base_url: &str) -> Spec<'_> {
        Spec { base_url, port: 7878, lifetime_s: 3600, terminate: TERMINATE }
    }

    /// Byte offset of `needle` in `text`, which must contain it.
    fn at(text: &str, needle: &str) -> usize {
        text.find(needle).unwrap_or_else(|| panic!("`{needle}` is missing from:\n{text}"))
    }

    #[test]
    fn stages_run_in_order_with_the_watchdog_first() {
        let s = script(&spec("https://github.com/himent12/folo/releases/download/v0.1.0/")).unwrap();
        assert!(s.starts_with("#!/bin/bash\n"), "{s}");
        let stages = [
            "set -euo pipefail",
            "BASE='https://github.com/himent12/folo/releases/download/v0.1.0'\n",
            "PORT=7878\n",
            TERMINATE,
            "( sleep 4200; log \"deadline reached\"; terminate_instance ) &",
            "trap on_exit EXIT",
            "apt-get install -y -q --no-install-recommends ca-certificates curl libvulkan1",
            "/etc/vulkan/icd.d/nvidia_icd.json <<'EOF_ICD'",
            "curl -fsSL --retry 3 --proto '=https,http' -o install.sh \"$BASE/install.sh\"",
            "sh install.sh --base-url \"$BASE\" --dir /usr/local/bin",
            "if has_gpu; then",
            "/usr/local/share/folo/nvidia_egl_icd.json <<'EOF_EGL'",
            "folo doctor --tensor-cores",
            "folo serve --host 0.0.0.0 --port \"$PORT\" --tensor-cores",
        ];
        let offsets: Vec<usize> = stages.iter().map(|n| at(&s, n)).collect();
        assert!(offsets.windows(2).all(|w| w[0] < w[1]), "stages out of order: {offsets:?}");
        // Before the watchdog only definitions and assignments run: no command that can fail.
        let before = &s[..at(&s, "( sleep 4200")];
        for risky in ["apt-get update", "curl -fsSL", "mktemp", "cd "] {
            assert!(!before.contains(risky), "`{risky}` runs before the watchdog");
        }
        // `folo serve` is the last command and runs in the foreground (the EXIT trap waits for it).
        assert!(s.trim_end().ends_with("--tensor-cores"), "{s}");
    }

    #[test]
    fn the_script_holds_no_secret_and_reads_the_token_from_the_environment() {
        let s = script(&spec("https://example.com/folo")).unwrap();
        assert!(s.contains("${FOLO_TOKEN:-}"));
        assert!(!s.contains("--token"), "the token must not be on folo serve's command line");
    }

    #[test]
    fn a_hostile_url_stays_one_quoted_word() {
        let hostile = "https://evil.example/'$(touch${IFS}/tmp/pwned)'`id`";
        let s = script(&spec(hostile)).unwrap();
        let line = s.lines().find(|l| l.starts_with("BASE=")).unwrap();
        assert_eq!(line, r#"BASE='https://evil.example/'\''$(touch${IFS}/tmp/pwned)'\''`id`'"#);
        assert_eq!(quote("a'b"), r#"'a'\''b'"#);
    }

    #[test]
    fn urls_that_are_not_downloads_are_refused() {
        for bad in [
            "",
            "ftp://x",
            "https://",
            "https:///path",
            "https://:8080/folo",
            "https://[]/folo",
            "https://[::1/folo",
            "https://[::1]junk/folo",
            "https://[not-ipv6]/folo",
            "https://example.com:abc/folo",
            "https://example.com:0/folo",
            "https://example.com:99999/folo",
            "https://example.com:/folo",
            "https://user:pass@example.com/folo",
            "https://example.com/folo?token=secret",
            "https://example.com/folo#frag",
            "file:///srv/folo",
            "https://a b",
            "https://a\nb",
        ] {
            let e = script(&spec(bad)).unwrap_err();
            assert_eq!(e.kind, ErrorKind::Invalid, "{bad:?}");
            assert!(e.message.contains("worker_base_url"), "{}", e.message);
        }
        assert!(check_base_url("http://10.0.0.5:8000/folo").is_ok());
        assert!(check_base_url("http://[::1]:8000/folo").is_ok());
    }

    #[test]
    fn the_gpu_check_falls_back_to_the_egl_driver() {
        let s = script(&spec("https://example.com/folo")).unwrap();
        // The probe reads folo's own answer (no pipe into grep: with pipefail, grep closing early fails it).
        assert!(s.contains(r#"doctor="$(folo doctor --json 2>/dev/null || true)""#), "{s}");
        assert!(s.contains(r#"*'"vendor":4318'*) return 0"#), "{s}");
        // Both spellings of the loader's variable, set only for the fallback and dropped when it does not help.
        let fallback = &s[at(
            &s,
            "EOF_EGL
  #",
        )..at(&s, "# 6. The worker.")];
        for line in [
            "export VK_DRIVER_FILES=/usr/local/share/folo/nvidia_egl_icd.json",
            "export VK_ICD_FILENAMES=/usr/local/share/folo/nvidia_egl_icd.json",
            "unset VK_DRIVER_FILES VK_ICD_FILENAMES",
        ] {
            assert!(
                fallback.contains(line),
                "`{line}` is missing from:
{fallback}"
            );
        }
        assert!(s.contains(r#""library_path": "libEGL_nvidia.so.0""#), "{s}");
    }

    #[test]
    fn the_time_it_keeps_trying_to_terminate_is_the_constant() {
        let s = script(&spec("https://example.com/folo")).unwrap();
        let tries = &s[at(&s, "terminate_instance() {")..at(&s, "# 1. The deadline watchdog")];
        assert!(tries.contains("for attempt in 1 2 3 4 5; do") && tries.contains("sleep 30"), "{tries}");
        assert_eq!(TERMINATE_TRIES_S, 5 * 30);
    }

    #[test]
    fn the_watchdog_never_overflows() {
        let s = script(&Spec { lifetime_s: u64::MAX, ..spec("https://x.example") }).unwrap();
        assert!(s.contains(&format!("( sleep {};", u64::MAX)));
    }

    #[test]
    fn bash_parses_the_script() {
        let s = script(&spec("https://example.com/it's")).unwrap();
        match bash_syntax(&s) {
            None => eprintln!("skipped: no bash on PATH"),
            Some(r) => {
                assert_eq!(r, Ok(()), "{s}");
                // The check does look: a script cut inside the driver check's `if` fails it.
                let cut = &s[..at(&s, "elif ldconfig")];
                assert!(bash_syntax(cut).is_some_and(|r| r.is_err()));
            }
        }
    }
}
