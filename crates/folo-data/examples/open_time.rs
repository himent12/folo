//! Times opening a dataset (`folo_data::open_with`), for the scan cache's numbers (docs/DATA.md, Scan cache;
//! docs/hill/r2-trainer.md).
//!
//! ```sh
//! SCAN_CACHE=D:/scan-cache cargo run -p folo-data --release --example open_time -- D:/folo-data/coco [noverify]
//! ```
//!
//! `PATH` is anything `open_with` takes (a folder, a `data.yaml`, a COCO `.json`). `noverify` skips the image header
//! probes (`OpenOptions::verify_images = false`). With `SCAN_CACHE=DIR` the open uses the scan cache in `DIR`: run
//! twice for a cold and a warm number. Prints the time, the split sizes and the number of issues.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, verify) = match args.as_slice() {
        [path] => (path, true),
        [path, flag] if flag == "noverify" => (path, false),
        _ => {
            eprintln!("usage: open_time PATH [noverify]   (env SCAN_CACHE=DIR turns on the scan cache)");
            return ExitCode::from(2);
        }
    };
    if let Some(dir) = std::env::var_os("SCAN_CACHE").filter(|d| !d.is_empty()) {
        folo_data::set_scan_cache_dir(Some(dir.into()));
    }
    let t = std::time::Instant::now();
    let opts = folo_data::OpenOptions { verify_images: verify, ..Default::default() };
    let d = match folo_data::open_with(path, &opts) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open_time: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "open verify={verify}: {:.2} s, {:?} images (train, val, test), {} issues",
        t.elapsed().as_secs_f64(),
        d.split_sizes(),
        d.report.issues.len()
    );
    ExitCode::SUCCESS
}
