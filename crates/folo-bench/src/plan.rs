//! The catalog and plans (docs/API.md, *Vision benchmarks*: "A plan is ...").
//!
//! A plan names conditions of the catalog, each at most once, with 1 to 5 distinct severities in ascending order
//! and a weight in `(0, 10]`. [`check_plan`] enforces exactly that and normalises the free text; [`variants`]
//! lists what a plan runs.

use folo_api::{BENCH_SEVERITIES, BenchAxis, BenchAxisInfo, BenchCatalog, BenchPlan, BenchPlanCondition};

use crate::conditions::Condition;

/// The severities of every condition of the [`default_plan`].
pub const DEFAULT_SEVERITIES: [u8; 3] = [1, 3, 5];

/// The largest weight a plan's condition can have.
pub const MAX_WEIGHT: f32 = 10.0;

/// `GET /api/bench/catalog`: every condition, the severity count, the default plan and the five axes.
pub fn catalog() -> BenchCatalog {
    BenchCatalog {
        conditions: Condition::ALL.iter().map(|c| c.info()).collect(),
        severities: BENCH_SEVERITIES,
        default_plan: default_plan(),
        axes: BenchAxis::ALL
            .iter()
            .map(|&axis| BenchAxisInfo {
                axis,
                name: axis_name(axis).to_string(),
                weight: axis.weight(),
                description: axis_description(axis).to_string(),
            })
            .collect(),
    }
}

/// The display name of an axis, e.g. `Self-labeling`.
pub fn axis_name(axis: BenchAxis) -> &'static str {
    match axis {
        BenchAxis::Accuracy => "Accuracy",
        BenchAxis::Robustness => "Robustness",
        BenchAxis::SelfLabeling => "Self-labeling",
        BenchAxis::Calibration => "Calibration",
        BenchAxis::Localization => "Localization",
    }
}

/// What an axis measures, in a sentence (docs/BENCH.md, *The score*).
pub fn axis_description(axis: BenchAxis) -> &'static str {
    match axis {
        BenchAxis::Accuracy => {
            "Does it find the objects, with tight boxes? mAP@0.5:0.95 on the unchanged images."
        }
        BenchAxis::Robustness => {
            "How much of that accuracy survives the plan's conditions, as the weighted mean of the share kept."
        }
        BenchAxis::SelfLabeling => {
            "Could its own labels train a model? The share of all objects it labels while at least 9 in 10 of its \
             labels are right."
        }
        BenchAxis::Calibration => "Can you trust its confidence? One minus the expected calibration error.",
        BenchAxis::Localization => {
            "How tight are its correct boxes? Their mean IoU, rescaled from 0.5..1 to 0..1."
        }
    }
}

/// The plan of a benchmark that names none: every condition at severities 1, 3 and 5, weight 1.
pub fn default_plan() -> BenchPlan {
    BenchPlan {
        conditions: Condition::ALL
            .iter()
            .map(|c| BenchPlanCondition {
                condition: c.id().to_string(),
                severities: DEFAULT_SEVERITIES.to_vec(),
                weight: 1.0,
                why: None,
            })
            .collect(),
        rationale: None,
        designed_by: None,
    }
}

/// Trimmed text, `None` when empty.
fn text(s: &Option<String>) -> Option<String> {
    s.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

/// Checks a plan and returns it normalised (`why` and `rationale` trimmed, empty ones `None`; everything else as
/// given). Valid when every condition is in the catalog and named at most once, its severities are 1 to 5
/// distinct values in `1..=5` in ascending order, and its weight is a number with `0 < weight <= 10`. An empty
/// plan is valid. The error is one plain sentence naming the condition and the field (a `400 invalid_body`
/// message).
pub fn check_plan(plan: &BenchPlan) -> Result<BenchPlan, String> {
    let mut seen: Vec<Condition> = Vec::with_capacity(plan.conditions.len());
    let mut conditions = Vec::with_capacity(plan.conditions.len());
    for (i, pc) in plan.conditions.iter().enumerate() {
        let Some(c) = Condition::parse(&pc.condition) else {
            let known: Vec<&str> = Condition::ALL.iter().map(|c| c.id()).collect();
            return Err(format!(
                "plan condition {} is `{}`, which is not a condition of the catalog (one of {})",
                i + 1,
                pc.condition,
                known.join(", ")
            ));
        };
        if seen.contains(&c) {
            return Err(format!(
                "plan names the condition `{}` twice; name each condition at most once",
                c.id()
            ));
        }
        seen.push(c);
        let s = &pc.severities;
        let ascending = s.windows(2).all(|p| p[0] < p[1]);
        if s.is_empty() || !ascending || s.iter().any(|v| !(1..=BENCH_SEVERITIES).contains(v)) {
            return Err(format!(
                "the severities of `{}` are {s:?}; they must be 1 to {BENCH_SEVERITIES} distinct values from 1 to \
                 {BENCH_SEVERITIES}, in ascending order, e.g. [1, 3, 5]",
                c.id()
            ));
        }
        if !(pc.weight > 0.0 && pc.weight <= MAX_WEIGHT) {
            return Err(format!(
                "the weight of `{}` is {}; it must be above 0 and at most {MAX_WEIGHT}",
                c.id(),
                pc.weight
            ));
        }
        conditions.push(BenchPlanCondition {
            condition: pc.condition.clone(),
            severities: s.clone(),
            weight: pc.weight,
            why: text(&pc.why),
        });
    }
    Ok(BenchPlan { conditions, rationale: text(&plan.rationale), designed_by: plan.designed_by.clone() })
}

/// Every `(condition, severity)` a plan runs, in plan order (conditions as listed, severities as listed).
/// Conditions not in the catalog are skipped; pass a plan that passed [`check_plan`].
pub fn variants(plan: &BenchPlan) -> Vec<(Condition, u8)> {
    plan.conditions
        .iter()
        .filter_map(|pc| Condition::parse(&pc.condition).map(|c| (c, pc)))
        .flat_map(|(c, pc)| pc.severities.iter().map(move |&s| (c, s)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_api::{BenchDesigner, LabelAgentId};

    fn pc(condition: &str, severities: &[u8], weight: f32) -> BenchPlanCondition {
        BenchPlanCondition { condition: condition.into(), severities: severities.to_vec(), weight, why: None }
    }

    fn plan(conditions: Vec<BenchPlanCondition>) -> BenchPlan {
        BenchPlan { conditions, rationale: None, designed_by: None }
    }

    #[test]
    fn catalog_lists_everything() {
        let c = catalog();
        assert_eq!(c.conditions.len(), 12);
        assert_eq!(c.conditions[0].id, "low_light");
        assert_eq!(c.severities, 5);
        assert_eq!(c.axes.iter().map(|a| a.axis).collect::<Vec<_>>(), BenchAxis::ALL);
        assert!((c.axes.iter().map(|a| a.weight).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(c.default_plan, default_plan());
    }

    #[test]
    fn default_plan_is_valid_and_runs_36_variants() {
        let p = default_plan();
        assert_eq!(check_plan(&p).unwrap(), p);
        let v = variants(&p);
        assert_eq!(v.len(), 36);
        assert_eq!(
            &v[..4],
            &[
                (Condition::LowLight, 1),
                (Condition::LowLight, 3),
                (Condition::LowLight, 5),
                (Condition::Overexposure, 1)
            ]
        );
        assert!(check_plan(&plan(vec![])).is_ok());
        assert!(variants(&plan(vec![])).is_empty());
    }

    #[test]
    fn unknown_conditions_fail() {
        let e = check_plan(&plan(vec![pc("haze", &[1], 1.0), pc("fog", &[1], 1.0)])).unwrap_err();
        assert!(e.contains("plan condition 2 is `fog`") && e.contains("haze"), "{e}");
        assert!(check_plan(&plan(vec![pc("Haze", &[1], 1.0)])).is_err());
        assert!(check_plan(&plan(vec![pc("haze", &[1], 1.0)])).is_ok());
    }

    #[test]
    fn duplicates_fail() {
        let e = check_plan(&plan(vec![pc("jpeg", &[1], 1.0), pc("jpeg", &[3], 1.0)])).unwrap_err();
        assert!(e.contains("`jpeg` twice"), "{e}");
        assert!(check_plan(&plan(vec![pc("jpeg", &[1], 1.0), pc("haze", &[3], 1.0)])).is_ok());
    }

    #[test]
    fn severities_must_be_distinct_ascending_and_in_range() {
        for bad in [&[][..], &[0], &[6], &[3, 1], &[1, 1], &[1, 2, 3, 4, 5, 5], &[2, 9]] {
            let e = check_plan(&plan(vec![pc("defocus", bad, 1.0)])).unwrap_err();
            assert!(e.contains("severities of `defocus`"), "{bad:?}: {e}");
        }
        for good in [&[1][..], &[5], &[2, 4], &[1, 2, 3, 4, 5]] {
            assert!(check_plan(&plan(vec![pc("defocus", good, 1.0)])).is_ok(), "{good:?}");
        }
    }

    #[test]
    fn weights_must_be_in_range() {
        for bad in [0.0, -1.0, 10.001, f32::NAN, f32::INFINITY] {
            let e = check_plan(&plan(vec![pc("haze", &[1], bad)])).unwrap_err();
            assert!(e.contains("weight of `haze`"), "{bad}: {e}");
        }
        for good in [0.001, 1.0, 10.0] {
            assert!(check_plan(&plan(vec![pc("haze", &[1], good)])).is_ok(), "{good}");
        }
    }

    #[test]
    fn text_is_trimmed_and_blank_is_none() {
        let mut p = plan(vec![pc("haze", &[1, 3], 2.0), pc("jpeg", &[5], 1.0)]);
        p.conditions[0].why = Some("  murky pond  ".into());
        p.conditions[1].why = Some("   ".into());
        p.rationale = Some("\n".into());
        p.designed_by = Some(BenchDesigner { agent: LabelAgentId::Codex, model: None });
        let n = check_plan(&p).unwrap();
        assert_eq!(n.conditions[0].why.as_deref(), Some("murky pond"));
        assert_eq!(n.conditions[1].why, None);
        assert_eq!(n.rationale, None);
        assert_eq!(n.designed_by, p.designed_by);
        assert_eq!(variants(&n), vec![(Condition::Haze, 1), (Condition::Haze, 3), (Condition::Jpeg, 5)]);
    }
}
