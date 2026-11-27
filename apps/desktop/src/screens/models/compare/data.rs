//! Pure data helpers and deterministic RFC 4180 CSV + GFM Markdown exporters for side-by-side
//! model comparison (`Route::ModelCompare`).

use std::collections::HashMap;

use crate::api::{Head, Model, ModelEval};
use crate::arch::ModelArch;
use crate::screens::models::groups::{eval_set, eval_set_with_report, group_models};
use crate::store::Store;
use crate::ui::push_csv_safe_text;

/// Maximum number of models compared side by side.
pub const MODEL_COMPARE_MAX: usize = 4;

/// State owned by `FoloApp` for side-by-side model comparison (`Route::ModelCompare`).
#[derive(Clone, Debug, Default)]
pub struct ModelCompareState {
    /// Ordered model ids being compared (slot 0 is the Baseline).
    pub ids: Vec<String>,
}

/// Picks up to 2 sensible default models to compare: when `seed` is given, starts with `seed` and
/// pairs it with the highest-mAP model on the same evaluation set (else the next highest-mAP model);
/// otherwise picks the top 2 models in the largest ranked evaluation group (else the first 2 models).
pub fn default_compare_models(store: &Store, models: &[Model], seed: Option<&str>) -> Vec<String> {
    if models.is_empty() {
        return Vec::new();
    }
    let score = |m: &Model| m.metrics.as_ref().map_or(-1.0_f32, |x| x.map50_95);
    if let Some(sid) = seed
        && let Some(first) = models.iter().find(|m| m.id == sid)
    {
        let mut out = vec![first.id.clone()];
        let first_set = eval_set(store, first).key;
        let mut peers: Vec<&Model> = models.iter().filter(|m| m.id != sid).collect();
        peers.sort_by(|a, b| {
            let a_same = eval_set(store, a).key == first_set;
            let b_same = eval_set(store, b).key == first_set;
            b_same.cmp(&a_same).then(score(b).total_cmp(&score(a)))
        });
        if let Some(second) = peers.first() {
            out.push(second.id.clone());
        }
        return out;
    }
    let groups = group_models(store, models, Some(("map", true)));
    if let Some(g) = groups.iter().find(|g| g.set.ranked && g.rows.len() >= 2) {
        return g.rows.iter().take(2).map(|m| m.id.clone()).collect();
    }
    let mut ranked: Vec<&Model> = models.iter().collect();
    ranked.sort_by(|a, b| score(b).total_cmp(&score(a)));
    ranked.into_iter().take(2).map(|m| m.id.clone()).collect()
}

/// Drops deleted or duplicate ids, caps at [`MODEL_COMPARE_MAX`], and backfills up to 2 models
/// when fewer than 2 remain and the workspace has at least 2 models.
pub fn reconcile_compare_models(ids: &mut Vec<String>, store: &Store, models: &[Model]) {
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| models.iter().any(|m| m.id == *id) && seen.insert(id.clone()));
    ids.truncate(MODEL_COMPARE_MAX);
    if ids.len() < 2 && models.len() >= 2 {
        let seed = ids.first().map(String::as_str);
        for candidate in default_compare_models(store, models, seed) {
            if ids.len() >= 2 {
                break;
            }
            if !ids.contains(&candidate) {
                ids.push(candidate);
            }
        }
    }
}

/// Whether all compared models share the same ranked evaluation dataset and split, preferring each
/// loaded [`ModelEval`]'s recorded `dataset_id` + `split` provenance when present.
pub fn same_eval_set_with_evals(store: &Store, models: &[&Model], evals: &[Option<&ModelEval>]) -> bool {
    let Some(first) = models.first() else { return true };
    let first_eval = evals.first().copied().flatten();
    let base = eval_set_with_report(store, first, first_eval);
    if !base.ranked {
        return false;
    }
    models[1..].iter().enumerate().all(|(i, m)| {
        let ev = evals.get(i + 1).copied().flatten();
        let set = eval_set_with_report(store, m, ev);
        set.ranked && set.key == base.key
    })
}

/// One aligned per-class row across all compared models.
#[derive(Clone, Debug, PartialEq)]
pub struct PerClassDeltaRow {
    pub name: String,
    /// `(ap50_95, ap50)` for each compared model slot (`None` if the model doesn't include or hasn't
    /// evaluated this class).
    pub cells: Vec<Option<(f32, f32)>>,
    /// Signed `ap50_95 - baseline_ap50_95` per slot (`None` for slot 0 or if either side is missing).
    pub deltas_50_95: Vec<Option<f32>>,
    /// Signed `ap50 - baseline_ap50` per slot (`None` for slot 0 or if either side is missing).
    pub deltas_50: Vec<Option<f32>>,
}

/// Aligns per-class AP metrics across `models` by class name (baseline class order first, then any
/// extra classes from later models in first-seen order).
pub fn build_per_class_rows(models: &[&Model], evals: &[Option<&ModelEval>]) -> Vec<PerClassDeltaRow> {
    let mut class_order: Vec<String> = Vec::new();
    let mut per_model: Vec<HashMap<String, (f32, f32)>> = Vec::with_capacity(models.len());

    for (m, ev) in models.iter().zip(evals.iter()) {
        let mut map = HashMap::new();
        if let Some(ev) = ev {
            for c in &ev.per_class {
                let name =
                    m.classes.get(c.class as usize).cloned().unwrap_or_else(|| format!("class {}", c.class));
                if !class_order.contains(&name) {
                    class_order.push(name.clone());
                }
                map.insert(name, (c.map50_95, c.map50));
            }
        }
        per_model.push(map);
    }

    class_order
        .into_iter()
        .map(|name| {
            let cells: Vec<Option<(f32, f32)>> = per_model.iter().map(|m| m.get(&name).copied()).collect();
            let base = cells.first().copied().flatten();
            let deltas_50_95 = cells
                .iter()
                .enumerate()
                .map(|(i, c)| if i == 0 { None } else { Some((*c)?.0 - base?.0) })
                .collect();
            let deltas_50 = cells
                .iter()
                .enumerate()
                .map(|(i, c)| if i == 0 { None } else { Some((*c)?.1 - base?.1) })
                .collect();
            PerClassDeltaRow { name, cells, deltas_50_95, deltas_50 }
        })
        .collect()
}

/// Formats a signed 3-decimal delta (`+0.042`, `-0.018`, `0.000`).
pub fn fmt_signed_delta(delta: f32) -> String {
    if !delta.is_finite() {
        return String::new();
    }
    if delta.abs() < 0.0005 {
        "0.000".to_string()
    } else if delta > 0.0 {
        format!("+{delta:.3}")
    } else {
        format!("{delta:.3}")
    }
}

/// Formats a signed 2-decimal delta (`+0.04`, `-0.02`, `0.00`).
pub fn fmt_signed_delta2(delta: f32) -> String {
    if !delta.is_finite() {
        return String::new();
    }
    if delta.abs() < 0.005 {
        "0.00".to_string()
    } else if delta > 0.0 {
        format!("+{delta:.2}")
    } else {
        format!("{delta:.2}")
    }
}

fn slug_part(raw: &str, max_chars: usize) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in raw.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
        if slug.len() >= max_chars {
            break;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() { "model".into() } else { slug }
}

/// Suggested filename for a model comparison export (`<m0>-vs-<m1>-compare.<ext>`).
pub fn model_compare_filename(models: &[&Model], ext: &str) -> String {
    match models {
        [] => format!("models-compare.{ext}"),
        [a] => format!("{}-compare.{ext}", slug_part(&a.name, 40)),
        [a, b] => format!("{}-vs-{}-compare.{ext}", slug_part(&a.name, 28), slug_part(&b.name, 28)),
        [a, b, rest @ ..] => format!(
            "{}-vs-{}-plus-{}-compare.{ext}",
            slug_part(&a.name, 24),
            slug_part(&b.name, 24),
            rest.len()
        ),
    }
}

fn head_label(h: Head) -> &'static str {
    if h == Head::O2o { "end-to-end (o2o)" } else { "NMS (o2m)" }
}

enum CsvCell {
    Text(String),
    Number(String),
}

fn csv_row(out: &mut String, cols: &[CsvCell]) {
    for (i, col) in cols.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match col {
            CsvCell::Text(s) => push_csv_safe_text(out, s),
            CsvCell::Number(s) => out.push_str(s),
        }
    }
    out.push_str("\r\n");
}

/// Builds an RFC 4180 CRLF CSV comparing `models` side by side (summary metrics + per-class AP
/// deltas vs baseline slot 0). Every row has identical column count and formula-injection protection.
pub fn build_model_compare_csv(
    models: &[&Model],
    archs: &[ModelArch],
    eval_sets: &[String],
    evals: &[Option<&ModelEval>],
    same_set: bool,
) -> String {
    let mut out = String::new();
    if models.is_empty() {
        return out;
    }

    let mut header = vec![CsvCell::Text("section".into()), CsvCell::Text("metric".into())];
    for (i, m) in models.iter().enumerate() {
        if i == 0 {
            header.push(CsvCell::Text(format!("{} (baseline)", m.name)));
        } else {
            header.push(CsvCell::Text(m.name.clone()));
            header.push(CsvCell::Text(format!("{} delta vs baseline", m.name)));
        }
    }
    csv_row(&mut out, &header);

    let push_text = |out: &mut String, metric: &str, vals: Vec<String>| {
        let mut row = vec![CsvCell::Text("summary".into()), CsvCell::Text(metric.into())];
        for (i, v) in vals.into_iter().enumerate() {
            row.push(CsvCell::Text(v));
            if i > 0 {
                row.push(CsvCell::Text(String::new()));
            }
        }
        csv_row(out, &row);
    };

    let push_metric =
        |out: &mut String, section: &str, metric: &str, vals: Vec<Option<f32>>, decimals: usize| {
            let base = vals.first().copied().flatten();
            let mut row = vec![CsvCell::Text(section.into()), CsvCell::Text(metric.into())];
            for (i, v) in vals.into_iter().enumerate() {
                let formatted = v
                    .filter(|x| x.is_finite())
                    .map(|x| if decimals == 2 { format!("{x:.2}") } else { format!("{x:.3}") })
                    .unwrap_or_default();
                row.push(CsvCell::Number(formatted));
                if i > 0 {
                    let d = match (v.filter(|x| x.is_finite()), base.filter(|x| x.is_finite())) {
                        (Some(a), Some(b)) => {
                            if decimals == 2 {
                                fmt_signed_delta2(a - b)
                            } else {
                                fmt_signed_delta(a - b)
                            }
                        }
                        _ => String::new(),
                    };
                    row.push(CsvCell::Number(d));
                }
            }
            csv_row(out, &row);
        };

    push_text(&mut out, "Architecture", archs.iter().map(|a| a.as_str().to_string()).collect());
    push_text(&mut out, "Family", archs.iter().map(|a| a.family.label().to_string()).collect());
    push_text(&mut out, "Evaluation set", eval_sets.to_vec());
    if !same_set && models.len() > 1 {
        let mut caveat_row = vec![
            CsvCell::Text("summary".into()),
            CsvCell::Text("Evaluation caveat".into()),
            CsvCell::Text(
                "Models were evaluated on different datasets or unranked splits; compare deltas with caution."
                    .into(),
            ),
        ];
        while caveat_row.len() < header.len() {
            caveat_row.push(CsvCell::Text(String::new()));
        }
        csv_row(&mut out, &caveat_row);
    }
    push_text(&mut out, "Input size (px)", models.iter().map(|m| m.imgsz.to_string()).collect());
    push_text(&mut out, "Head", models.iter().map(|m| head_label(m.head).to_string()).collect());
    push_text(&mut out, "Parameters", models.iter().map(|m| m.params.to_string()).collect());
    push_text(&mut out, "GFLOPs", models.iter().map(|m| format!("{:.2}", m.flops_g)).collect());
    push_text(&mut out, "File size (bytes)", models.iter().map(|m| m.bytes.to_string()).collect());

    push_metric(
        &mut out,
        "summary",
        "mAP50-95",
        models.iter().map(|m| m.metrics.as_ref().map(|x| x.map50_95)).collect(),
        3,
    );
    push_metric(
        &mut out,
        "summary",
        "mAP50",
        models.iter().map(|m| m.metrics.as_ref().map(|x| x.map50)).collect(),
        3,
    );
    push_metric(
        &mut out,
        "summary",
        "Precision",
        models.iter().map(|m| m.metrics.as_ref().map(|x| x.precision)).collect(),
        3,
    );
    push_metric(
        &mut out,
        "summary",
        "Recall",
        models.iter().map(|m| m.metrics.as_ref().map(|x| x.recall)).collect(),
        3,
    );
    push_metric(
        &mut out,
        "summary",
        "Best F1 conf",
        models.iter().map(|m| m.metrics.as_ref().and_then(|x| x.best_f1_conf)).collect(),
        2,
    );

    let per_class = build_per_class_rows(models, evals);
    for row in &per_class {
        push_metric(
            &mut out,
            "per_class_ap50_95",
            &row.name,
            row.cells.iter().map(|c| c.map(|v| v.0)).collect(),
            3,
        );
    }
    for row in &per_class {
        push_metric(
            &mut out,
            "per_class_ap50",
            &row.name,
            row.cells.iter().map(|c| c.map(|v| v.1)).collect(),
            3,
        );
    }

    out
}

/// Escapes Markdown syntax characters and collapses newlines so arbitrary model, dataset, or class
/// names render literally in GitHub-Flavored Markdown prose and tables.
pub fn escape_md_cell(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\r' | '\n' => out.push(' '),
            '\\' | '|' | '[' | ']' | '(' | ')' | '`' | '*' | '_' | '<' | '>' | '~' | '!' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

fn md_row(out: &mut String, cols: &[String]) {
    out.push('|');
    for col in cols {
        out.push(' ');
        out.push_str(&escape_md_cell(col));
        out.push_str(" |");
    }
    out.push('\n');
}

fn md_sep(out: &mut String, count: usize) {
    out.push('|');
    for i in 0..count {
        if i == 0 {
            out.push_str(" --- |");
        } else {
            out.push_str(" ---: |");
        }
    }
    out.push('\n');
}

/// Builds a GitHub-Flavored Markdown comparison report (summary metrics table + per-class AP50-95
/// and AP50 delta tables vs baseline slot 0).
pub fn build_model_compare_markdown(
    models: &[&Model],
    archs: &[ModelArch],
    eval_sets: &[String],
    evals: &[Option<&ModelEval>],
    same_set: bool,
) -> String {
    let mut out = String::new();
    if models.is_empty() {
        return out;
    }

    out.push_str("# Model comparison\n\n");
    let names: Vec<String> = models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if i == 0 {
                format!("**{}** (baseline)", escape_md_cell(&m.name))
            } else {
                escape_md_cell(&m.name)
            }
        })
        .collect();
    out.push_str(&format!("Comparing {}\n", names.join(" vs ")));
    if !same_set && models.len() > 1 {
        out.push_str(
            "\n> Note: These models were evaluated on different datasets or unranked splits; compare accuracy metrics with caution.\n",
        );
    }

    out.push_str("\n## Summary metrics\n\n");
    let mut header = vec!["Metric".to_string()];
    for (i, m) in models.iter().enumerate() {
        if i == 0 {
            header.push(format!("{} (baseline)", m.name));
        } else {
            header.push(m.name.clone());
            header.push(format!("{} Δ", m.name));
        }
    }
    let col_count = header.len();
    md_row(&mut out, &header);
    md_sep(&mut out, col_count);

    let push_text = |out: &mut String, metric: &str, vals: Vec<String>| {
        let mut row = vec![metric.to_string()];
        for (i, v) in vals.into_iter().enumerate() {
            row.push(v);
            if i > 0 {
                row.push("—".into());
            }
        }
        md_row(out, &row);
    };

    let push_metric = |out: &mut String, metric: &str, vals: Vec<Option<f32>>, decimals: usize| {
        let base = vals.first().copied().flatten();
        let mut row = vec![metric.to_string()];
        for (i, v) in vals.into_iter().enumerate() {
            let formatted = v
                .filter(|x| x.is_finite())
                .map(|x| if decimals == 2 { format!("{x:.2}") } else { format!("{x:.3}") })
                .unwrap_or_else(|| "—".into());
            row.push(formatted);
            if i > 0 {
                let d = match (v.filter(|x| x.is_finite()), base.filter(|x| x.is_finite())) {
                    (Some(a), Some(b)) => {
                        if decimals == 2 {
                            fmt_signed_delta2(a - b)
                        } else {
                            fmt_signed_delta(a - b)
                        }
                    }
                    _ => "—".into(),
                };
                row.push(d);
            }
        }
        md_row(out, &row);
    };

    push_text(&mut out, "Architecture", archs.iter().map(|a| a.as_str().to_string()).collect());
    push_text(&mut out, "Family", archs.iter().map(|a| a.family.label().to_string()).collect());
    push_text(&mut out, "Evaluation set", eval_sets.to_vec());
    push_text(&mut out, "Input size", models.iter().map(|m| format!("{} px", m.imgsz)).collect());
    push_text(&mut out, "Head", models.iter().map(|m| head_label(m.head).to_string()).collect());
    push_text(
        &mut out,
        "Parameters",
        models
            .iter()
            .map(|m| {
                let (v, u) = crate::ui::count_si(m.params);
                format!("{v} {u}")
            })
            .collect(),
    );
    push_text(&mut out, "Compute", models.iter().map(|m| format!("{:.1} GFLOPs", m.flops_g)).collect());
    push_text(
        &mut out,
        "File size",
        models
            .iter()
            .map(|m| {
                let (v, u) = crate::ui::bytes(m.bytes);
                format!("{v} {u}")
            })
            .collect(),
    );

    push_metric(
        &mut out,
        "mAP50-95",
        models.iter().map(|m| m.metrics.as_ref().map(|x| x.map50_95)).collect(),
        3,
    );
    push_metric(&mut out, "mAP50", models.iter().map(|m| m.metrics.as_ref().map(|x| x.map50)).collect(), 3);
    push_metric(
        &mut out,
        "Precision",
        models.iter().map(|m| m.metrics.as_ref().map(|x| x.precision)).collect(),
        3,
    );
    push_metric(&mut out, "Recall", models.iter().map(|m| m.metrics.as_ref().map(|x| x.recall)).collect(), 3);
    push_metric(
        &mut out,
        "Best F1 conf",
        models.iter().map(|m| m.metrics.as_ref().and_then(|x| x.best_f1_conf)).collect(),
        2,
    );

    let per_class = build_per_class_rows(models, evals);
    if !per_class.is_empty() {
        out.push_str("\n## Per-class AP50-95\n\n");
        let mut pc_hdr = vec!["Class".to_string()];
        for (i, m) in models.iter().enumerate() {
            if i == 0 {
                pc_hdr.push(format!("{} (baseline)", m.name));
            } else {
                pc_hdr.push(m.name.clone());
                pc_hdr.push(format!("{} Δ", m.name));
            }
        }
        md_row(&mut out, &pc_hdr);
        md_sep(&mut out, pc_hdr.len());
        for row in &per_class {
            push_metric(&mut out, &row.name, row.cells.iter().map(|c| c.map(|v| v.0)).collect(), 3);
        }

        out.push_str("\n## Per-class AP50\n\n");
        md_row(&mut out, &pc_hdr);
        md_sep(&mut out, pc_hdr.len());
        for row in &per_class {
            push_metric(&mut out, &row.name, row.cells.iter().map(|c| c.map(|v| v.1)).collect(), 3);
        }
    }

    out
}
