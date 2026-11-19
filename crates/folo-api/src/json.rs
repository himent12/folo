//! JSON presentation helpers shared by the CLI's `--json` records and the MCP results: floats without f32 → f64
//! noise and paths with `/` separators (quoted where needed in a command the user may paste: [`cli_command`]).

use std::path::Path;

use serde::Serialize;
use serde_json::Value;

/// `v` as a JSON value whose floats read as they were stored ([`clean_floats`]).
pub fn to_value<T: Serialize + ?Sized>(v: &T) -> serde_json::Result<Value> {
    let mut v = serde_json::to_value(v)?;
    clean_floats(&mut v);
    Ok(v)
}

/// Rewrites every number that is an `f32` widened to `f64` as the shortest decimal of that `f32`, so
/// `0.009999999776482582` (the `f32` 0.01 through `serde_json::to_value` or `json!`) reads `0.01` again and a box
/// coordinate `207.92527770996094` reads `207.92528`. Numbers that are not exactly an `f32` (genuine `f64`s such
/// as durations) are left alone; one that is changes by less than half an `f32` step.
pub fn clean_floats(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            if let Some(x) = n.as_f64()
                && let Some(clean) = shortest_f32(x)
                && let Some(num) = serde_json::Number::from_f64(clean)
            {
                *n = num;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(clean_floats),
        Value::Object(o) => o.values_mut().for_each(clean_floats),
        _ => {}
    }
}

fn shortest_f32(x: f64) -> Option<f64> {
    let f = x as f32;
    if !f.is_finite() || f as f64 != x {
        return None;
    }
    let clean: f64 = f.to_string().parse().ok()?;
    (clean != x).then_some(clean)
}

/// A path as FOLO reports it in JSON: `/` separators (Windows accepts them) and no `\\?\` verbatim prefix.
pub fn path(p: impl AsRef<Path>) -> String {
    let s = p.as_ref().to_string_lossy();
    let s = s
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .unwrap_or_else(|| s.strip_prefix(r"\\?\").unwrap_or(&s).to_string());
    if cfg!(windows) { s.replace('\\', "/") } else { s }
}

/// A command FOLO prints for the user to paste (`folo train --resume {checkpoint}`): each `{name}` in `template` is
/// that name's path in `paths` ([`path`]), bare when it holds only ASCII letters, digits and `-_./:`, else in double
/// quotes, which PowerShell, cmd and POSIX shells all read as one argument. A path that one of those shells changes
/// inside double quotes (`UNQUOTABLE`, or a control character) is never put in the command: `<name>` stands in its
/// place, and a sentence after the command gives the path as it is, for the user to quote for their shell. A `{name}`
/// without a path also prints `<name>`. `template` is FOLO's own text: a path in it could hold a `{name}`.
pub fn cli_command(template: &str, paths: &[(&str, &Path)]) -> String {
    let (mut line, mut notes, mut rest) = (String::new(), String::new(), template);
    while let Some((before, after)) = rest.split_once('{')
        && let Some((name, after)) = after.split_once('}')
    {
        line.push_str(before);
        let p = paths.iter().find(|(n, _)| *n == name).map(|(_, p)| path(p));
        match p {
            Some(p)
                if !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:".contains(c)) =>
            {
                line.push_str(&p)
            }
            Some(p) if !p.chars().any(|c| UNQUOTABLE.contains(c) || c.is_control()) => {
                line.push_str(&format!("\"{p}\""))
            }
            p => {
                line.push_str(&format!("<{name}>"));
                if let Some(p) = p {
                    notes.push_str(&format!(
                        ". In place of <{name}>, paste this path quoted for your shell: {p}"
                    ));
                }
            }
        }
        rest = after;
    }
    line + rest + &notes
}

/// What PowerShell (`"` `$` `` ` `` and the typographic double quotes, which it reads as `"`), cmd (`%`, `!` with
/// delayed expansion) or a POSIX shell (`$` `` ` `` `\`, `!` when interactive) changes inside double quotes. On
/// Windows [`path`] has turned every `\` into `/`.
const UNQUOTABLE: &str = "\"$`%!\\\u{201C}\u{201D}\u{201E}\u{201F}\u{2033}";

/// A duration in milliseconds as FOLO reports it: rounded to 0.1 µs, so a sum of stage times reads `10.5163` rather
/// than `10.516300000000001`. Timers are not that precise anyway.
pub fn round_ms(ms: f64) -> f64 {
    if ms.is_finite() { (ms * 1e4).round() / 1e4 } else { ms }
}

/// `#[serde(serialize_with = ...)]` for millisecond fields: [`round_ms`].
pub fn serialize_ms<S: serde::Serializer>(ms: &f64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(round_ms(*ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn milliseconds_print_without_f64_noise() {
        assert_eq!(round_ms(10.516300000000001), 10.5163);
        assert_eq!(round_ms(0.6174000000000001), 0.6174);
        assert_eq!(round_ms(81.48469999999999), 81.4847);
        assert_eq!(round_ms(0.000_04), 0.0);
        assert!(round_ms(f64::NAN).is_nan());
        let t = crate::Timing {
            preprocess_ms: 10.5163,
            inference_ms: 3.3925,
            postprocess_ms: 0.1 + 0.2,
            total_ms: 0.0,
        };
        let t = crate::Timing { total_ms: t.preprocess_ms + t.inference_ms + t.postprocess_ms, ..t };
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            r#"{"preprocess_ms":10.5163,"inference_ms":3.3925,"postprocess_ms":0.3,"total_ms":14.2088}"#
        );
        let v = crate::ValTiming { total_s: 0.4, per_image_ms: 4.3 * 3.0 };
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"{"total_s":0.4,"per_image_ms":12.9}"#);
    }

    #[test]
    fn floats_lose_their_f32_noise() {
        let v = to_value(
            &json!({ "lr0": 0.01f32, "hsv": [0.015f32, 0.7f32], "x": 207.925_28f32, "half": 0.5f32 }),
        )
        .unwrap();
        assert_eq!(v.to_string(), r#"{"lr0":0.01,"hsv":[0.015,0.7],"x":207.92528,"half":0.5}"#);
        // Genuine f64s and integers are untouched.
        let v = to_value(&json!({ "s": 31.2f64, "third": 1.0f64 / 3.0, "n": 7, "big": 1e300 })).unwrap();
        assert_eq!(v["s"], 31.2);
        assert_eq!(v["third"], 1.0f64 / 3.0);
        assert_eq!(v["n"], 7);
        assert_eq!(v["big"], 1e300);
    }

    #[test]
    fn paths_use_forward_slashes() {
        assert_eq!(path("runs/a/weights/best.folo"), "runs/a/weights/best.folo");
        if cfg!(windows) {
            assert_eq!(path(r"runs\shapes-n\weights/best.folo"), "runs/shapes-n/weights/best.folo");
            assert_eq!(path(r"\\?\D:\data\x"), "D:/data/x");
            assert_eq!(path(r"\\?\UNC\server\share\x"), "//server/share/x");
        }
    }

    #[test]
    fn command_paths_are_quoted_when_they_need_it() {
        let cmd = |p: &str| cli_command("folo dataset stats {data}", &[("data", Path::new(p))]);
        let arg = |p: &str| cmd(p).strip_prefix("folo dataset stats ").unwrap().to_string();
        for bare in ["D:/data/shapes-n_2.folo", "C:/", "//server/share/x", "runs/a.b:c"] {
            assert_eq!(arg(bare), bare);
        }
        for (p, quoted) in [
            ("runs/run with space/stop café", "\"runs/run with space/stop café\""),
            ("D:/a,b;c&d(e)#f~g=h", "\"D:/a,b;c&d(e)#f~g=h\""),
            ("'lead/it's", "\"'lead/it's\""),
            ("//server/share/a b", "\"//server/share/a b\""),
            ("{data}/x", "\"{data}/x\""),
            ("", "\"\""),
        ] {
            assert_eq!(arg(p), quoted);
        }
        if cfg!(windows) {
            assert_eq!(arg(r"C:\Users\a b\x"), "\"C:/Users/a b/x\"");
            assert_eq!(arg(r"\\server\share\a b"), "\"//server/share/a b\"");
            assert_eq!(arg(r"C:\"), "C:/");
        }
        // A path a shell would change inside double quotes is never in the command.
        let unquotable =
            ['"', '$', '`', '%', '!', '\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}', '\u{2033}', '\n', '\t'];
        for c in unquotable.into_iter().chain((!cfg!(windows)).then_some('\\')) {
            let p = format!("D:/a{c}PROBE{c}b");
            assert_eq!(
                cmd(&p),
                format!(
                    "folo dataset stats <data>. In place of <data>, paste this path quoted for your shell: {p}"
                ),
                "{c:?}"
            );
        }
    }

    #[test]
    fn a_whole_hint_with_one_unquotable_path() {
        let hint = |model: &str, data: Option<&str>| {
            let mut paths = vec![("model", Path::new(model))];
            paths.extend(data.map(|d| ("dataset", Path::new(d))));
            cli_command("start a new run: folo train --model {model} --data {dataset} --epochs <n>", &paths)
        };
        assert_eq!(
            hint("runs/a$b/weights/best.folo", Some("D:/data/my set")),
            "start a new run: folo train --model <model> --data \"D:/data/my set\" --epochs <n>. In place of \
             <model>, paste this path quoted for your shell: runs/a$b/weights/best.folo"
        );
        assert_eq!(
            hint("runs/a/weights/best.folo", None),
            "start a new run: folo train --model runs/a/weights/best.folo --data <dataset> --epochs <n>"
        );
    }
}
