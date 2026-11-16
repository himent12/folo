//! A final layout cached before its node was measured again is not reused. taffy's block algorithm
//! lays a block's children out at the size it measures with, so answering the next final-layout
//! request from that cache entry left them there: FOLO Desktop's run page, beside a log with long
//! wrapped lines, drew its chart cards stretched across their row with their content at the cards'
//! shrink-to-fit width (the charts' widths then followed the legends' numbers under the cursor).

use taffy::{AvailableSpace as Space, LayoutInput, Line, RequestedAxis, RunMode, SizingMode};

use super::*;
use crate::window::layout_tree::LayoutTree;
use crate::{Length, Style, relative};

fn request(run_mode: RunMode, width: Option<f32>, available: Space) -> LayoutInput {
    LayoutInput {
        run_mode,
        sizing_mode: SizingMode::InherentSize,
        axis: RequestedAxis::Both,
        known_dimensions: taffy::Size { width, height: None },
        parent_size: taffy::Size { width, height: None },
        available_space: taffy::Size { width: available, height: Space::MaxContent },
        vertical_margins_are_collapsible: Line::FALSE,
    }
}

#[test]
fn measuring_a_block_after_its_final_layout_does_not_leave_its_children_at_the_measured_width() {
    let (mut app, handle) = open(size(px(400.), px(300.)), 1.0, |_, _| crate::div().into_any_element());
    app.update(|cx| {
        cx.update_window(handle, |_, window, cx| {
            // A block card whose content is `w_full`, 10 px tall.
            let content = Style {
                size: crate::Size { width: relative(1.).into(), height: Length::from(px(10.)) },
                ..Style::default()
            };
            let content = window.request_layout(content, [], cx);
            let card = window.request_layout(Style::default(), [content], cx);
            let mut engine = window.layout_engine.take().expect("the layout engine");
            let tree = &mut engine.tree;
            let width = |tree: &LayoutTree| tree.unrounded(content.0).size.width;
            // The card's final layout at 800 px, then a min-content measure (as an ancestor's
            // intrinsic sizing makes), which lays the content out at the card's 0 px min-content
            // width, then the same final layout again.
            tree.request(
                card.0,
                request(RunMode::PerformLayout, Some(800.), Space::Definite(800.)),
                window,
                cx,
            );
            assert_eq!(width(tree), 800.);
            tree.request(card.0, request(RunMode::ComputeSize, None, Space::MinContent), window, cx);
            assert_eq!(width(tree), 0.);
            tree.request(
                card.0,
                request(RunMode::PerformLayout, Some(800.), Space::Definite(800.)),
                window,
                cx,
            );
            assert_eq!(width(tree), 800., "the content is laid out at the card's width again");
            window.layout_engine = Some(engine);
        })
    })
    .expect("window");
}
