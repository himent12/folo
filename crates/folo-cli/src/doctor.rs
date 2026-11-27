//! `folo doctor`: the host, its GPU adapters and the one FOLO picks, and the Studio `folo serve` would serve.

use std::path::PathBuf;

use folo_cli::VERSION;
use folo_server::WebSource;
use folo_server::system::CpuSimd;
use serde_json::json;

use crate::cli::DoctorArgs;
use crate::output::{CliError, Out};
use crate::serve::{DEFAULT_WEB_DIR, resolve_web};

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

#[derive(Debug, Clone, Copy, Default)]
struct SimdOverrides {
    scalar: bool,
    no_vnni: bool,
    no_avx512: bool,
}

impl SimdOverrides {
    fn from_env() -> Self {
        Self {
            scalar: std::env::var_os("FOLO_Q8_SCALAR").is_some(),
            no_vnni: std::env::var_os("FOLO_Q8_NO_VNNI").is_some(),
            no_avx512: std::env::var_os("FOLO_Q8_NO_AVX512").is_some(),
        }
    }

    fn active_names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.scalar {
            names.push("FOLO_Q8_SCALAR");
        }
        if self.no_vnni {
            names.push("FOLO_Q8_NO_VNNI");
        }
        if self.no_avx512 {
            names.push("FOLO_Q8_NO_AVX512");
        }
        names
    }
}

fn format_simd(simd: &CpuSimd, overrides: SimdOverrides) -> String {
    let mut isas = Vec::new();
    if simd.avx2 {
        isas.push("avx2");
    }
    if simd.avx_vnni {
        isas.push("avx-vnni");
    }
    if simd.avx512_vnni {
        isas.push("avx512-vnni");
    }
    if simd.neon {
        isas.push("neon");
    }
    let isa = if isas.is_empty() { "none".to_string() } else { isas.join(", ") };
    let mut notes = vec![format!("int8: {}", simd.int8_kernel)];
    let active = overrides.active_names();
    if !active.is_empty() {
        let verb = if simd.int8_kernel == "portable" && (simd.avx2 || simd.avx_vnni || simd.avx512_vnni) {
            "disabled by"
        } else {
            "override:"
        };
        notes.push(format!("{verb} {}", active.join(", ")));
    }
    format!("{isa} ({})", notes.join("; "))
}

fn abs_path(p: PathBuf) -> PathBuf {
    std::path::absolute(&p).unwrap_or(p)
}

fn non_empty_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn resolved_home_dir() -> PathBuf {
    if let Some(h) = non_empty_env("FOLO_HOME") {
        return h;
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    home.map_or_else(|| PathBuf::from(".folo"), |h| PathBuf::from(h).join(".folo"))
}

fn resolved_data_dir() -> PathBuf {
    non_empty_env("FOLO_DATA_DIR").unwrap_or_else(resolved_home_dir)
}

fn resolved_weights_dir() -> PathBuf {
    non_empty_env("FOLO_WEIGHTS_DIR").unwrap_or_else(|| resolved_home_dir().join("weights"))
}

fn resolved_cache_dir() -> PathBuf {
    non_empty_env("FOLO_CACHE_DIR").unwrap_or_else(|| resolved_home_dir().join("cache"))
}

pub(crate) fn doctor(args: DoctorArgs, out: Out) -> Result<(), CliError> {
    let host = folo_server::system::host_info();
    let simd = folo_server::system::cpu_simd();
    let simd_overrides = SimdOverrides::from_env();
    let data_dir = abs_path(resolved_data_dir());
    let weights_dir = abs_path(resolved_weights_dir());
    let cache_dir = abs_path(resolved_cache_dir());
    let cache_disabled = std::env::var_os("FOLO_NO_CACHE").is_some_and(|v| !v.is_empty() && v != "0");
    let web_dir = args.web_dir.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_WEB_DIR));
    let web_exists = web_dir.is_dir();
    let web_ready = folo_server::web_dir_ready(&web_dir);
    let web_wasm = folo_server::web::dir_has_folo_wasm(&web_dir);
    let embedded = folo_server::web::embedded();
    let embedded_wasm = embedded.is_some_and(folo_server::web::has_folo_wasm);
    let embedded = embedded.map(<[_]>::len);
    let web_source = match resolve_web(args.web_dir.clone()) {
        WebSource::Embedded => "embedded",
        _ => "dir",
    };
    // Whether the Studio `folo serve` would serve can run models in the browser (its Playground's WebGPU / CPU).
    // (`folo serve` does not serve a folder without index.html at all.)
    let served_wasm = if web_source == "embedded" { embedded_wasm } else { web_ready && web_wasm };
    if args.tensor_cores {
        folo_gpu::context::request_tensor_cores();
    }
    let options = folo_gpu::GpuOptions::from_env();
    let adapters = folo_gpu::GpuContext::enumerate(&options);
    // Creating the device (the shared one, as `folo serve` does for `device: auto`) proves the default adapter
    // works and shows which one FOLO picks.
    let capabilities = folo_server::system::capabilities();
    let selected = folo_gpu::GpuContext::shared().map(|c| (c.summary().clone(), c.tensor_cores_status()));
    let gpus: Vec<_> = adapters
        .iter()
        .map(|a| {
            json!({
                "index": a.index,
                "name": a.name,
                "backend": a.backend.to_ascii_lowercase(),
                "device_type": a.device_type,
                "vendor": a.vendor,
                "device": a.device,
                "driver": a.driver,
                "driver_info": a.driver_info,
                "shader_f16": a.shader_f16,
                "tensor_cores": a.tensor_cores,
                "memory_bytes": a.memory_bytes,
                "max_buffer_bytes": a.max_buffer_size,
                "max_storage_binding_bytes": a.max_storage_buffer_binding_size,
                "workgroup_storage_bytes": a.max_compute_workgroup_storage_size,
                "max_invocations_per_workgroup": a.max_compute_invocations_per_workgroup,
                "subgroup_size": [a.subgroup_size.0, a.subgroup_size.1],
            })
        })
        .collect();
    let selected_json = match &selected {
        Ok((s, tc)) => json!({
            "index": s.index,
            "name": s.name,
            "backend": s.backend.to_ascii_lowercase(),
            "tensor_cores": tc,
        }),
        Err(e) => json!({ "error": e.to_string() }),
    };
    let report = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "git_sha": env!("FOLO_GIT_SHA"),
        "build_date": env!("FOLO_BUILD_DATE"),
        "os": host.os,
        "arch": host.arch,
        "cpu": {
            "name": host.cpu_name,
            "threads": host.cpu_threads,
            "simd": {
                "avx2": simd.avx2,
                "avx_vnni": simd.avx_vnni,
                "avx512_vnni": simd.avx512_vnni,
                "neon": simd.neon,
                "int8_fast": simd.int8_fast,
                "int8_kernel": simd.int8_kernel,
            },
        },
        "memory": { "total_bytes": host.memory_total_bytes, "available_bytes": host.memory_available_bytes },
        "paths": {
            "data_dir": crate::output::record_path(&data_dir),
            "weights_dir": crate::output::record_path(&weights_dir),
            "cache_dir": crate::output::record_path(&cache_dir),
            "cache_enabled": !cache_disabled,
        },
        "gpus": gpus,
        "gpu_detection": "wgpu",
        "gpu_selected": selected_json,
        "capabilities": capabilities,
        "web_dist": { "path": crate::output::record_path(&web_dir), "exists": web_exists, "index_html": web_ready, "folo_wasm": web_wasm },
        "web_embedded": { "present": embedded.is_some(), "files": embedded.unwrap_or(0), "folo_wasm": embedded_wasm },
        "web_source": web_source,
        "web_folo_wasm": served_wasm,
    });
    out.record(&report);
    out.human(format!("folo      {VERSION}"));
    out.human(format!("os        {} ({})", host.os, host.arch));
    out.human(format!("cpu       {} ({} threads)", host.cpu_name, host.cpu_threads));
    out.human(format!("simd      {}", format_simd(&simd, simd_overrides)));
    out.human(format!(
        "memory    {} total, {} available",
        gib(host.memory_total_bytes),
        gib(host.memory_available_bytes)
    ));
    out.human(format!("data      {}", crate::output::record_path(&data_dir)));
    out.human(format!("weights   {}", crate::output::record_path(&weights_dir)));
    out.human(if cache_disabled {
        format!("cache     {} (disabled by FOLO_NO_CACHE)", crate::output::record_path(&cache_dir))
    } else {
        format!("cache     {}", crate::output::record_path(&cache_dir))
    });
    if adapters.is_empty() {
        out.human("gpu       no adapters found (Vulkan / DX12 / Metal); FOLO runs on the CPU");
    }
    for a in &adapters {
        out.human(format!("gpu       {a}"));
    }
    match &selected {
        Ok((s, tc)) => {
            out.human(format!(
                "gpu       FOLO uses [{}] {} ({}); set {} to pick another",
                s.index,
                s.name,
                s.backend,
                folo_gpu::context::ENV_ADAPTER
            ));
            out.human(format!("gpu       tensor cores (f16): {tc}"));
        }
        Err(e) => out.human(format!("gpu       no usable adapter: {e}")),
    }
    let tc = &capabilities.tensor_cores;
    let amp = match &capabilities.amp.reason {
        None => format!("mixed-precision training (amp) available on {}", capabilities.device),
        // The adapter has them, only this process did not ask: `folo train --amp` does.
        Some(_)
            if tc.available
                && tc.reason.as_deref() == Some(folo_gpu::context::TENSOR_CORES_NOT_REQUESTED) =>
        {
            format!(
                "available on {}: `folo train --amp` asks for tensor cores itself; `folo serve` and `folo mcp` need \
                 `--tensor-cores` for amp runs",
                capabilities.device
            )
        }
        Some(why) => format!("not available: {why}"),
    };
    out.human(format!("amp       {amp}"));
    let web = if web_ready {
        "built".to_string()
    } else if web_exists {
        "folder exists but has no index.html".to_string()
    } else {
        "not built (run `npm ci && npm run build` in web/)".to_string()
    };
    out.human(format!("web/dist  {} — {web}", folo_api::json::path(&web_dir)));
    let studio = match embedded {
        Some(n) if web_source == "embedded" => format!("embedded ({n} files); `folo serve` uses it"),
        Some(n) => format!("embedded ({n} files); `folo serve --web-dir` overrides it"),
        None => "not embedded (build with --features embed-web)".to_string(),
    };
    out.human(format!("studio    {studio}"));
    let source = if web_source == "embedded" {
        "the embedded Studio".to_string()
    } else {
        folo_api::json::path(&web_dir)
    };
    let wasm = if served_wasm {
        format!("folo-wasm in {source}; the Playground can run models in the browser (WebGPU or CPU)")
    } else {
        format!(
            "no folo-wasm in {source}; the Playground runs models on the server only (build it with scripts/build-wasm before `npm run build`)"
        )
    };
    out.human(format!("browser   {wasm}"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_simd_describes_isa_and_int8_status() {
        let none = SimdOverrides::default();
        assert_eq!(
            format_simd(
                &CpuSimd {
                    avx2: true,
                    avx_vnni: true,
                    avx512_vnni: true,
                    neon: false,
                    int8_fast: true,
                    int8_kernel: "AVX-512 VNNI (512-bit)".into(),
                },
                none
            ),
            "avx2, avx-vnni, avx512-vnni (int8: AVX-512 VNNI (512-bit))"
        );
        assert_eq!(
            format_simd(
                &CpuSimd {
                    avx2: true,
                    avx_vnni: true,
                    avx512_vnni: true,
                    neon: false,
                    int8_fast: true,
                    int8_kernel: "AVX-VNNI".into(),
                },
                SimdOverrides { no_avx512: true, ..none }
            ),
            "avx2, avx-vnni, avx512-vnni (int8: AVX-VNNI; override: FOLO_Q8_NO_AVX512)"
        );
        assert_eq!(
            format_simd(
                &CpuSimd {
                    avx2: true,
                    avx_vnni: true,
                    avx512_vnni: false,
                    neon: false,
                    int8_fast: false,
                    int8_kernel: "portable".into(),
                },
                SimdOverrides { scalar: true, ..none }
            ),
            "avx2, avx-vnni (int8: portable; disabled by FOLO_Q8_SCALAR)"
        );
        assert_eq!(
            format_simd(
                &CpuSimd {
                    avx2: true,
                    avx_vnni: false,
                    avx512_vnni: false,
                    neon: false,
                    int8_fast: false,
                    int8_kernel: "AVX2".into(),
                },
                SimdOverrides { no_vnni: true, ..none }
            ),
            "avx2 (int8: AVX2; override: FOLO_Q8_NO_VNNI)"
        );
        assert_eq!(
            format_simd(
                &CpuSimd {
                    avx2: false,
                    avx_vnni: false,
                    avx512_vnni: false,
                    neon: true,
                    int8_fast: false,
                    int8_kernel: "portable".into(),
                },
                none
            ),
            "neon (int8: portable)"
        );
        assert_eq!(
            format_simd(
                &CpuSimd {
                    avx2: false,
                    avx_vnni: false,
                    avx512_vnni: false,
                    neon: false,
                    int8_fast: false,
                    int8_kernel: "portable".into(),
                },
                none
            ),
            "none (int8: portable)"
        );
    }
}
