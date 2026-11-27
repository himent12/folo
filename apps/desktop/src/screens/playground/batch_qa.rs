//! Playground Batch QA filter (`All` / `With detections` / `Empty` / `By class`) with live match
//! counts, filtered previous/next stepping, and compact chip strip UI.

use std::collections::{BTreeMap, BTreeSet};

use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use super::batch::{BatchItemStatus, PlayBatchState};
use super::page::filter_detections;
use crate::app::{FoloApp, act};
use crate::theme::Theme;

/// Maximum per-class QA filter chips displayed in the filmstrip before truncating (any actively
/// selected class chip is always kept visible).
const MAX_QA_CLASS_CHIPS: usize = 8;

/// Active QA filter on the Playground batch filmstrip and Batch Summary CSV export.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BatchQaFilter {
    #[default]
    All,
    /// Completed fresh images with at least one detection above the active confidence/class filter.
    WithDetections,
    /// Completed fresh images with zero detections above the active confidence/class filter.
    Empty,
    /// Completed fresh images containing at least one detection of `class_id` above `conf`.
    ByClass(u32),
}

/// Live match counts for the Playground batch QA filter strip.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchQaCounts {
    pub all: usize,
    pub with_detections: usize,
    pub empty: usize,
    /// `(class_id, class_name, matching_image_count)` sorted by `matching_image_count DESC, class_id ASC`.
    pub by_class: Vec<(u32, String, usize)>,
}

impl PlayBatchState {
    /// True when batch item `idx` matches `filter` under the current Playground settings.
    #[allow(clippy::too_many_arguments)]
    pub fn item_matches_qa(
        &self,
        idx: usize,
        filter: BatchQaFilter,
        model_id: Option<&str>,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        play_class: Option<u32>,
        e2e: bool,
    ) -> bool {
        if filter == BatchQaFilter::All {
            return idx < self.items.len();
        }
        let Some(mid) = model_id else { return false };
        let Some(res) = self.fresh_result(idx, mid, iou, target, conf, e2e) else {
            return false;
        };
        match filter {
            BatchQaFilter::All => true,
            BatchQaFilter::WithDetections => !filter_detections(&res.pred.boxes, conf, play_class).is_empty(),
            BatchQaFilter::Empty => filter_detections(&res.pred.boxes, conf, play_class).is_empty(),
            BatchQaFilter::ByClass(cid) => !filter_detections(&res.pred.boxes, conf, Some(cid)).is_empty(),
        }
    }

    /// Indices into `self.items` matching `self.qa_filter` under the current Playground settings.
    pub fn filtered_indices(
        &self,
        model_id: Option<&str>,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        play_class: Option<u32>,
        e2e: bool,
    ) -> Vec<usize> {
        (0..self.items.len())
            .filter(|&i| {
                self.item_matches_qa(i, self.qa_filter, model_id, iou, target, conf, play_class, e2e)
            })
            .collect()
    }

    /// Computes live match counts for `All`, `With detections`, `Empty`, and each detected class
    /// across fresh completed batch items.
    pub fn qa_counts(
        &self,
        model_id: Option<&str>,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        play_class: Option<u32>,
        e2e: bool,
        model_classes: &[String],
    ) -> BatchQaCounts {
        let mut counts = BatchQaCounts { all: self.items.len(), ..Default::default() };
        let mut class_imgs: BTreeMap<u32, (String, usize)> = BTreeMap::new();
        for item in &self.items {
            let BatchItemStatus::Done(res) = &item.status else { continue };
            if !model_id.is_some_and(|m| res.matches_current(m, iou, target, conf, e2e)) {
                continue;
            }
            if filter_detections(&res.pred.boxes, conf, play_class).is_empty() {
                counts.empty += 1;
            } else {
                counts.with_detections += 1;
            }
            let mut seen_in_image: BTreeSet<u32> = BTreeSet::new();
            for b in res.pred.boxes.iter().filter(|b| b.score >= conf) {
                if seen_in_image.insert(b.class) {
                    let fallback = || {
                        model_classes
                            .get(b.class as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("class {}", b.class))
                    };
                    let name = if b.name.trim().is_empty() { fallback() } else { b.name.clone() };
                    class_imgs.entry(b.class).and_modify(|(_, c)| *c += 1).or_insert_with(|| (name, 1));
                }
            }
        }
        if let BatchQaFilter::ByClass(cid) = self.qa_filter
            && !class_imgs.contains_key(&cid)
        {
            let name = model_classes.get(cid as usize).cloned().unwrap_or_else(|| format!("class {cid}"));
            class_imgs.insert(cid, (name, 0));
        }
        let mut by_class: Vec<(u32, String, usize)> =
            class_imgs.into_iter().map(|(cid, (name, n))| (cid, name, n)).collect();
        by_class.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        counts.by_class = by_class;
        counts
    }

    /// Computes the next item index in `filtered` when stepping by `delta` (`-1` or `+1`).
    pub fn next_filtered_index(&self, filtered: &[usize], delta: i32) -> Option<usize> {
        if filtered.is_empty() {
            return None;
        }
        if let Some(pos) = filtered.iter().position(|&i| i == self.active) {
            if filtered.len() == 1 {
                return None;
            }
            let next = ((pos as i32 + delta).rem_euclid(filtered.len() as i32)) as usize;
            return Some(filtered[next]);
        }
        if delta >= 0 {
            filtered.iter().copied().find(|&i| i > self.active).or_else(|| filtered.first().copied())
        } else {
            filtered.iter().copied().rev().find(|&i| i < self.active).or_else(|| filtered.last().copied())
        }
    }
}

impl FoloApp {
    /// Returns the currently filtered Playground batch indices under `play_batch.qa_filter`.
    pub(crate) fn play_batch_filtered_indices(&self) -> Vec<usize> {
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        self.play_batch.filtered_indices(
            self.play_model.as_deref(),
            self.play_iou,
            self.play_target,
            self.play_conf,
            self.play_class,
            e2e,
        )
    }

    /// Compact QA filter chip row rendered inside the Playground batch filmstrip.
    pub(super) fn play_batch_qa_filter_row(
        &self,
        _t: &Theme,
        counts: &BatchQaCounts,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let cur = self.play_batch.qa_filter;
        let mut chips: Vec<AnyElement> =
            Vec::with_capacity(3 + counts.by_class.len().min(MAX_QA_CLASS_CHIPS));

        chips.push(
            self.chip(
                "play-batch-qa-all",
                "All",
                cur == BatchQaFilter::All,
                None,
                Some(counts.all as u64),
                window,
                cx,
                act(|this, _, cx| this.set_play_batch_qa_filter(BatchQaFilter::All, cx)),
            )
            .into_any_element(),
        );
        chips.push(
            self.chip(
                "play-batch-qa-det",
                "With detections",
                cur == BatchQaFilter::WithDetections,
                None,
                Some(counts.with_detections as u64),
                window,
                cx,
                act(|this, _, cx| {
                    let next = if this.play_batch.qa_filter == BatchQaFilter::WithDetections {
                        BatchQaFilter::All
                    } else {
                        BatchQaFilter::WithDetections
                    };
                    this.set_play_batch_qa_filter(next, cx);
                }),
            )
            .into_any_element(),
        );
        chips.push(
            self.chip(
                "play-batch-qa-empty",
                "Empty",
                cur == BatchQaFilter::Empty,
                None,
                Some(counts.empty as u64),
                window,
                cx,
                act(|this, _, cx| {
                    let next = if this.play_batch.qa_filter == BatchQaFilter::Empty {
                        BatchQaFilter::All
                    } else {
                        BatchQaFilter::Empty
                    };
                    this.set_play_batch_qa_filter(next, cx);
                }),
            )
            .into_any_element(),
        );

        let mut shown_classes: Vec<(u32, String, usize)> = counts
            .by_class
            .iter()
            .take(MAX_QA_CLASS_CHIPS)
            .map(|(cid, name, n)| (*cid, name.clone(), *n))
            .collect();
        if let BatchQaFilter::ByClass(active_cid) = cur
            && !shown_classes.iter().any(|(cid, _, _)| *cid == active_cid)
            && let Some((cid, name, n)) = counts.by_class.iter().find(|(cid, _, _)| *cid == active_cid)
        {
            shown_classes.push((*cid, name.clone(), *n));
        }
        for (cid, cname, n) in shown_classes {
            let active = cur == BatchQaFilter::ByClass(cid);
            chips.push(
                self.chip(
                    &format!("play-batch-qa-cls-{cid}"),
                    cname,
                    active,
                    Some(cid),
                    Some(n as u64),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let next = if this.play_batch.qa_filter == BatchQaFilter::ByClass(cid) {
                            BatchQaFilter::All
                        } else {
                            BatchQaFilter::ByClass(cid)
                        };
                        this.set_play_batch_qa_filter(next, cx);
                    }),
                )
                .into_any_element(),
            );
        }

        div().flex().flex_wrap().items_center().gap(px(4.)).children(chips).into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ImageSize, PredictResponse, ScoredBox, Timing};
    use crate::screens::playground::{BatchItemResult, PlayBatchItem, PlayBatchSource};

    fn done_item(name: &str, boxes: Vec<ScoredBox>) -> PlayBatchItem {
        PlayBatchItem {
            name: name.to_string().into(),
            source: PlayBatchSource::Dataset { dataset_id: "ds".into(), image_id: name.into() },
            status: BatchItemStatus::Done(BatchItemResult {
                pred: PredictResponse {
                    model_id: "m_1".into(),
                    image: ImageSize { width: 640, height: 480 },
                    boxes,
                    timing: Timing {
                        preprocess_ms: 1.0,
                        inference_ms: 4.0,
                        postprocess_ms: 0.5,
                        total_ms: 5.5,
                    },
                    device: "cpu".into(),
                    cold: false,
                },
                width: 640,
                height: 480,
                model_id: "m_1".into(),
                iou_bits: 0.7_f32.to_bits(),
                target: None,
                floor: 0.05,
                content_hash: None,
            }),
        }
    }

    #[test]
    fn qa_filter_counts_indices_and_stepping_behave_accurately() {
        let mut st = PlayBatchState {
            items: vec![
                done_item(
                    "a.jpg",
                    vec![ScoredBox {
                        class: 0,
                        name: "circle".into(),
                        score: 0.9,
                        x1: 10.,
                        y1: 10.,
                        x2: 50.,
                        y2: 50.,
                    }],
                ),
                done_item("b.jpg", vec![]),
                done_item(
                    "c.jpg",
                    vec![
                        ScoredBox {
                            class: 0,
                            name: "circle".into(),
                            score: 0.85,
                            x1: 10.,
                            y1: 10.,
                            x2: 50.,
                            y2: 50.,
                        },
                        ScoredBox {
                            class: 1,
                            name: "square".into(),
                            score: 0.75,
                            x1: 60.,
                            y1: 10.,
                            x2: 100.,
                            y2: 50.,
                        },
                    ],
                ),
                PlayBatchItem {
                    name: "d.jpg".into(),
                    source: PlayBatchSource::Dataset { dataset_id: "ds".into(), image_id: "d".into() },
                    status: BatchItemStatus::Pending,
                },
            ],
            active: 0,
            ..Default::default()
        };

        let classes = vec!["circle".into(), "square".into()];
        let counts = st.qa_counts(Some("m_1"), 0.7, None, 0.25, None, false, &classes);
        assert_eq!(counts.all, 4);
        assert_eq!(counts.with_detections, 2);
        assert_eq!(counts.empty, 1);
        assert_eq!(counts.by_class, vec![(0, "circle".into(), 2), (1, "square".into(), 1)]);

        st.qa_filter = BatchQaFilter::WithDetections;
        let dets = st.filtered_indices(Some("m_1"), 0.7, None, 0.25, None, false);
        assert_eq!(dets, vec![0, 2]);
        assert_eq!(st.next_filtered_index(&dets, 1), Some(2));
        assert_eq!(st.next_filtered_index(&dets, -1), Some(2));

        // When active is outside the filtered list (e.g. index 1), +1 steps to next > active (2), -1 to 0.
        st.active = 1;
        assert_eq!(st.next_filtered_index(&dets, 1), Some(2));
        assert_eq!(st.next_filtered_index(&dets, -1), Some(0));

        st.qa_filter = BatchQaFilter::Empty;
        assert_eq!(st.filtered_indices(Some("m_1"), 0.7, None, 0.25, None, false), vec![1]);

        st.qa_filter = BatchQaFilter::ByClass(1);
        assert_eq!(st.filtered_indices(Some("m_1"), 0.7, None, 0.25, None, false), vec![2]);
    }
}
