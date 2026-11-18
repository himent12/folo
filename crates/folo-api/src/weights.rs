//! Published pretrained weights: the names `folo weights pull` downloads from a GitHub release, and the local cache
//! they land in. Shared by the CLI (`folo weights`, `train --weights <name>`) and the server, which lists the
//! downloaded ones as models (`Model.source: published`, the id is the name), so a published name works in
//! `TrainConfig.weights`, `/api/predict`, MCP and Studio as it does on the command line.

use std::path::PathBuf;

/// The release asset listing `<sha256>  <file>` for every weights file of a release.
pub const SUMS: &str = "SHA256SUMS";

/// A published weights file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Published {
    /// What users type: `folo weights pull <name>`, `--weights <name>`; also its model id on a server.
    pub name: &'static str,
    /// Release tag.
    pub release: &'static str,
    /// Release asset (also the cached file name).
    pub asset: &'static str,
    /// One line for `folo weights list` and `Model.about`.
    pub about: &'static str,
}

/// Every published weights file.
pub const PUBLISHED: &[Published] = &[
    Published {
        name: "folo-n-coco",
        release: "weights-v1",
        asset: "folo-n-coco.folo",
        about: "folo-n trained from scratch on COCO train2017 at 640 px (EMA weights, f32, unfused: fine-tune or \
                predict); examples/coco",
    },
    Published {
        name: "folo-n-coco-f16",
        release: "weights-v1",
        asset: "folo-n-coco-f16.folo",
        about: "folo-n-coco fused and cast to f16 (folo export): inference only",
    },
    Published {
        name: "folo-n-coco-v2",
        release: "weights-v2",
        asset: "folo-n-coco-v2.folo",
        about: "folo-n from scratch on COCO train2017, interim: epoch 62 of a paused 300-epoch run, val2017 AP 0.317 \
                (EMA weights, f32, unfused: fine-tune or predict); examples/coco",
    },
    Published {
        name: "folo-n-coco-v2-f16",
        release: "weights-v2",
        asset: "folo-n-coco-v2-f16.folo",
        about: "folo-n-coco-v2 fused and cast to f16 (folo export): inference only",
    },
    Published {
        name: "folo-s-coco",
        release: "weights-v3",
        asset: "folo-s-coco.folo",
        about: "folo-s trained from scratch on COCO train2017 at 640 px, 56 epochs, val2017 AP 0.416 (EMA weights, \
                f32, unfused: fine-tune or predict); examples/coco",
    },
    Published {
        name: "folo-s-coco-f16",
        release: "weights-v3",
        asset: "folo-s-coco-f16.folo",
        about: "folo-s-coco fused and cast to f16 (folo export): inference only",
    },
    Published {
        name: "folo-n-coco-v3",
        release: "weights-v4",
        asset: "folo-n-coco-v3.folo",
        about: "folo-n from scratch on COCO train2017: folo-n-coco-v2 annealed for 16 more epochs (78 in all), \
                val2017 AP 0.340 (EMA weights, f32, unfused: fine-tune or predict); examples/coco",
    },
    Published {
        name: "folo-n-coco-v3-f16",
        release: "weights-v4",
        asset: "folo-n-coco-v3-f16.folo",
        about: "folo-n-coco-v3 fused and cast to f16 (folo export): inference only",
    },
    Published {
        name: "folo-n-coco-v4",
        release: "weights-v5",
        asset: "folo-n-coco-v4.folo",
        about: "folo-n from scratch on COCO train2017, 64 epochs, the second half distilled from folo-s-coco, \
                val2017 AP 0.356 (EMA weights, f32, unfused: fine-tune or predict); examples/coco",
    },
    Published {
        name: "folo-n-coco-v4-f16",
        release: "weights-v5",
        asset: "folo-n-coco-v4-f16.folo",
        about: "folo-n-coco-v4 fused and cast to f16 (folo export): inference only",
    },
    Published {
        name: "folo-n-coco-v5",
        release: "weights-v6",
        asset: "folo-n-coco-v5.folo",
        about: "folo-n-coco-v4 continued for 64 epochs on COCO train2017 with the folo-s-coco teacher, \
                val2017 AP 0.370 (EMA weights, f32, unfused: fine-tune or predict); examples/coco",
    },
    Published {
        name: "folo-n-coco-v5-f16",
        release: "weights-v6",
        asset: "folo-n-coco-v5-f16.folo",
        about: "folo-n-coco-v5 fused and cast to f16 (folo export): inference only",
    },
];

/// The published weights called `name`.
pub fn find(name: &str) -> Option<&'static Published> {
    PUBLISHED.iter().find(|p| p.name == name)
}

/// The published names, comma-separated (for hints).
pub fn names() -> String {
    PUBLISHED.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
}

/// FOLO's home folder: `$FOLO_HOME`, else `~/.folo` (`%USERPROFILE%\.folo` on Windows), else `.folo`.
pub fn home_dir() -> PathBuf {
    if let Some(h) = std::env::var_os("FOLO_HOME") {
        return PathBuf::from(h);
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    home.map_or_else(|| PathBuf::from(".folo"), |h| PathBuf::from(h).join(".folo"))
}

/// The cache folder `folo weights pull` downloads into: `$FOLO_WEIGHTS_DIR`, else `<home_dir>/weights`. A server
/// reads it at start-up, whatever its `--data-dir`.
pub fn cache_dir() -> PathBuf {
    match std::env::var_os("FOLO_WEIGHTS_DIR") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => home_dir().join("weights"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_cached_by_asset() {
        for (i, a) in PUBLISHED.iter().enumerate() {
            assert!(a.asset.ends_with(".folo"));
            // Usable as model ids: no dots (derived files are `<id>.<hash>.<variant>`), no path separators.
            assert!(a.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'), "{}", a.name);
            assert!(!a.name.starts_with("m_"), "model ids of runs and files start with m_");
            assert!(PUBLISHED[i + 1..].iter().all(|b| b.name != a.name && b.asset != a.asset));
        }
        assert_eq!(find("folo-n-coco").map(|p| p.asset), Some("folo-n-coco.folo"));
        assert!(names().contains("folo-n-coco-f16"));
        assert_eq!(find("folo-n-coco-v2").map(|p| p.release), Some("weights-v2"));
        assert_eq!(
            find("folo-s-coco").map(|p| (p.release, p.asset)),
            Some(("weights-v3", "folo-s-coco.folo"))
        );
        assert_eq!(find("folo-s-coco-f16").map(|p| p.asset), Some("folo-s-coco-f16.folo"));
        assert_eq!(
            find("folo-n-coco-v3").map(|p| (p.release, p.asset)),
            Some(("weights-v4", "folo-n-coco-v3.folo"))
        );
        assert_eq!(
            find("folo-n-coco-v3-f16").map(|p| (p.release, p.asset)),
            Some(("weights-v4", "folo-n-coco-v3-f16.folo"))
        );
        assert_eq!(
            find("folo-n-coco-v4").map(|p| (p.release, p.asset)),
            Some(("weights-v5", "folo-n-coco-v4.folo"))
        );
        assert_eq!(
            find("folo-n-coco-v4-f16").map(|p| (p.release, p.asset)),
            Some(("weights-v5", "folo-n-coco-v4-f16.folo"))
        );
        assert_eq!(
            find("folo-n-coco-v5").map(|p| (p.release, p.asset)),
            Some(("weights-v6", "folo-n-coco-v5.folo"))
        );
        assert_eq!(
            find("folo-n-coco-v5-f16").map(|p| (p.release, p.asset)),
            Some(("weights-v6", "folo-n-coco-v5-f16.folo"))
        );
        assert!(names().ends_with("folo-n-coco-v4, folo-n-coco-v4-f16, folo-n-coco-v5, folo-n-coco-v5-f16"));
    }
}
