//! `folo mcp`: the MCP server (folo-mcp) on stdio, embedded on the data dir, on the mock, or proxying a
//! `folo serve`.

use crate::cli::McpArgs;
use crate::output::{CliError, ExitCode};
use crate::serve::{data_dir_in_use, data_dir_lock_failed, init_tracing};
use crate::{data_dir, default_data_dir, serve_lock};

/// `folo mcp`: MCP over stdio. stdout belongs to the protocol, so nothing else is printed there (not even with
/// `--json`); logs go to stderr.
pub(crate) fn mcp(args: McpArgs) -> Result<(), CliError> {
    init_tracing();
    // Validate `--tls-ca` upfront (before touching the data dir or acquiring `serve.lock`) so an unreadable or
    // unusable CA bundle fails immediately instead of falling through to `data_dir_in_use` or embedded mode.
    let tls_ca = serve_lock::load_tls_ca(args.tls_ca.as_deref())?;
    if args.tensor_cores && args.server.is_none() {
        // As `folo serve --tensor-cores`: before anything creates the GPU context.
        folo_gpu::context::request_tensor_cores();
    }
    // Embedded mode writes there; a mock or proxying server does not.
    let data_dir = if args.server.is_none() && !args.mock {
        data_dir(args.data_dir.clone())?
    } else {
        args.data_dir.clone().unwrap_or_else(default_data_dir)
    };
    let mut backend = match (&args.server, args.mock) {
        (Some(url), _) => {
            folo_mcp::Backend::Proxy { url: url.clone(), token: args.token.clone(), tls_ca: tls_ca.clone() }
        }
        (None, true) => folo_mcp::Backend::Mock,
        (None, false) => folo_mcp::Backend::Real,
    };
    // Embedded, it owns the data dir as `folo serve` does. When a `folo serve` (FOLO Desktop's, say) already
    // owns it, it goes through that server instead (not with --confine: the server reads datasets unconfined).
    let _data_lock = match backend {
        folo_mcp::Backend::Real => match serve_lock::acquire(&data_dir) {
            Ok(serve_lock::Acquire::Held(lock)) => Some(lock),
            Ok(serve_lock::Acquire::InUse { url: Some(url) })
                if !args.confine
                    && serve_lock::serves(&url, &data_dir, args.token.as_deref(), tls_ca.as_ref()) =>
            {
                tracing::info!(%url, "a folo serve owns the data dir: using it (as --server)");
                backend = folo_mcp::Backend::Proxy { url, token: args.token.clone(), tls_ca: tls_ca.clone() };
                None
            }
            Ok(serve_lock::Acquire::InUse { url }) => return Err(data_dir_in_use(&data_dir, url.as_deref())),
            Err(e) => return Err(data_dir_lock_failed(&data_dir, e)),
        },
        _ => None,
    };
    let opts = folo_mcp::Options {
        data_dir: data_dir.clone(),
        read_only: args.read_only,
        confine: args.confine,
        device: args.device.clone(),
        published_weights: Some(folo_api::weights::cache_dir()),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::new(ExitCode::Runtime, "runtime", format!("could not start tokio: {e}")))?;
    let result = runtime.block_on(async move {
        let server = folo_mcp::build(backend.clone(), opts)
            .map_err(|e| CliError::new(ExitCode::Usage, "usage", e).hint("see `folo mcp --help`"))?;
        tracing::info!(?backend, data_dir = %folo_api::json::path(&data_dir), read_only = args.read_only, confine = args.confine, "folo mcp on stdio");
        folo_mcp::serve_stdio(server, folo_server::shutdown_signal())
            .await
            .map_err(|e| CliError::new(ExitCode::Runtime, "mcp_failed", e))
    });
    // tokio's stdin reader sits on a blocking thread that cannot be interrupted; do not wait for it.
    runtime.shutdown_background();
    result
}
