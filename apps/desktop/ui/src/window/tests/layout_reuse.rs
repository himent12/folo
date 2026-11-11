//! The layout tree reuses last frame's results for unchanged subtrees (`window/layout_tree.rs`). Its
//! contract is exactness: a window that has drawn any sequence of frames shows, byte for byte, what a new
//! window drawing the same tree for the first time shows. Random trees (flex rows, columns and wrapping,
//! a grid, wrapping, clamped and truncated text) are changed at random between frames and checked.

use std::{cell::RefCell, rc::Rc};

use super::open;
use crate::AppContext as _;
use crate::{
    AnyElement, Div, HeadlessApp, IntoElement, ParentElement, SharedString, Styled, div, hsla, px, relative,
    size,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const WORDS: [&str; 8] = ["run", "loss", "0.6123", "epoch", "validation", "mAP50-95", "·", "shapes-hard"];

/// A node of the random tree.
#[derive(Clone)]
enum Node {
    Text { words: Vec<usize>, nowrap: bool, truncate: bool, clamp: bool },
    Box { kind: u8, width: u8, grow: bool, padding: u8, gap: u8, center: bool, children: Vec<Node> },
}

fn random_node(rng: &mut Rng, depth: u32) -> Node {
    if depth == 0 || rng.below(3) == 0 {
        let words = (0..1 + rng.below(6)).map(|_| rng.below(WORDS.len() as u64) as usize).collect();
        return Node::Text {
            words,
            nowrap: rng.below(3) == 0,
            truncate: rng.below(3) == 0,
            clamp: rng.below(4) == 0,
        };
    }
    let children = (0..1 + rng.below(4)).map(|_| random_node(rng, depth - 1)).collect();
    Node::Box {
        kind: rng.below(4) as u8,
        width: rng.below(4) as u8,
        grow: rng.below(2) == 0,
        padding: rng.below(3) as u8,
        gap: rng.below(3) as u8,
        center: rng.below(2) == 0,
        children,
    }
}

/// Changes one random thing (a text, a style, a child added or removed), or nothing.
fn mutate(rng: &mut Rng, node: &mut Node) {
    match node {
        Node::Text { words, nowrap, truncate, .. } => match rng.below(4) {
            0 => words.push(rng.below(WORDS.len() as u64) as usize),
            1 if words.len() > 1 => {
                words.pop();
            }
            2 => *nowrap = !*nowrap,
            _ => *truncate = !*truncate,
        },
        Node::Box { children, kind, width, grow, .. } => {
            if !children.is_empty() && rng.below(3) != 0 {
                let ix = rng.below(children.len() as u64) as usize;
                mutate(rng, &mut children[ix]);
                return;
            }
            match rng.below(6) {
                0 => children.push(random_node(rng, 2)),
                1 if children.len() > 1 => {
                    children.remove(rng.below(children.len() as u64) as usize);
                }
                2 => *kind = rng.below(4) as u8,
                3 => *width = rng.below(4) as u8,
                4 => *grow = !*grow,
                _ => {}
            }
        }
    }
}

fn element(node: &Node, ix: &mut usize) -> AnyElement {
    *ix += 1;
    let color = hsla((*ix as f32 * 0.13) % 1., 0.6, 0.6, 1.);
    match node {
        Node::Text { words, nowrap, truncate, clamp } => {
            let text: String = words.iter().map(|w| WORDS[*w]).collect::<Vec<_>>().join(" ");
            let mut el = div().text_color(hsla(0., 0., 0., 1.)).min_w_0();
            if *nowrap {
                el = el.whitespace_nowrap();
            }
            if *truncate {
                el = el.text_ellipsis();
            }
            if *clamp {
                el = el.line_clamp(2);
            }
            el.child(SharedString::from(text)).into_any_element()
        }
        Node::Box { kind, width, grow, padding, gap, center, children } => {
            let mut el: Div = div().bg(color).p(px(*padding as f32 * 3.)).gap(px(*gap as f32 * 2.));
            el = match kind {
                0 => el.flex(),
                1 => el.flex().flex_col(),
                2 => el.flex().flex_wrap(),
                _ => el.grid().grid_cols(2),
            };
            el = match width {
                0 => el,
                1 => el.w(px(70.)),
                2 => el.w(relative(0.5)),
                _ => el.min_w_0().overflow_hidden(),
            };
            if *grow {
                el = el.flex_1();
            }
            if *center {
                el = el.items_center();
            }
            el.children(children.iter().map(|child| element(child, ix))).into_any_element()
        }
    }
}

/// A window drawing `model` (its width is part of the model, so the root's request changes too).
fn window(model: Rc<RefCell<(Node, f32)>>) -> (HeadlessApp, crate::AnyWindowHandle) {
    open(size(px(260.), px(200.)), 1.5, move |_, _| {
        let (node, width) = &*model.borrow();
        div().w(px(*width)).flex().flex_col().child(element(node, &mut 0)).into_any_element()
    })
}

#[test]
fn a_reused_layout_is_the_layout_of_a_new_window() {
    for seed in 1..=24u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let model = Rc::new(RefCell::new((random_node(&mut rng, 4), 240.)));
        let (mut app, handle) = window(model.clone());
        for step in 0..12 {
            {
                let mut m = model.borrow_mut();
                match rng.below(5) {
                    0 => m.1 = 120. + rng.below(140) as f32,
                    1 => {}
                    _ => mutate(&mut rng, &mut m.0),
                }
            }
            app.update(|cx| cx.update_window(handle, |_, window, _| window.refresh())).expect("window");
            let reused = app.frame(handle).clone();
            let (mut fresh, fresh_handle) = window(Rc::new(RefCell::new(model.borrow().clone())));
            assert!(fresh.frame(fresh_handle) == &reused, "seed {seed}, step {step}");
        }
    }
}

#[test]
fn an_unchanged_frame_lays_out_nothing_again() {
    let mut rng = Rng(7);
    let model = Rc::new(RefCell::new((random_node(&mut rng, 4), 240.)));
    let (mut app, handle) = window(model);
    app.update(|cx| cx.update_window(handle, |_, window, _| window.refresh())).expect("window");
    app.frame(handle);
    let (answered, computed) = app.read(|cx| match cx.windows.get(&handle.id) {
        Some(Some(window)) => {
            let tree = &window.layout_engine.as_ref().expect("engine").tree;
            (tree.answered_last, tree.computed_last)
        }
        _ => (0, 0),
    });
    assert!(answered == 1 && computed == 0, "the root answers from its records: {answered} / {computed}");
}

/// Lays out a root with two measured leaves through the window's engine and returns every node's bounds
/// and the leaves' measure calls (`layouts`: the available widths of each `compute_layout`).
fn lay_out(window: &mut crate::Window, cx: &mut crate::App, layouts: &[f32]) -> (Vec<String>, Vec<String>) {
    use crate::{AvailableSpace, Style, window::MeasureKey};
    let calls: Rc<RefCell<Vec<String>>> = Rc::default();
    let mut leaves = Vec::new();
    for (ix, text) in ["alpha", "beta gamma"].into_iter().enumerate() {
        let calls = calls.clone();
        let key = MeasureKey { hash: ix as u64, text: text.into() };
        leaves.push(window.request_keyed_measured_layout(
            Style::default(),
            key,
            || {},
            move |known, available, _, _| {
                calls.borrow_mut().push(format!("{ix} {known:?} {available:?}"));
                let width = match available.width {
                    AvailableSpace::Definite(w) => w.min(px(text.len() as f32 * 7.)),
                    _ => px(text.len() as f32 * 7.),
                };
                size(width, px(10. * (text.len() as f32 * 7. / width.0).ceil()))
            },
        ));
    }
    let mut style = Style::default();
    style.display = crate::Display::Flex;
    let root = window.request_layout(style, leaves.iter().copied(), cx);
    for width in layouts {
        let space = size(AvailableSpace::Definite(px(*width)), AvailableSpace::MinContent);
        window.compute_layout(root, space, cx);
    }
    let bounds =
        std::iter::once(root).chain(leaves).map(|id| format!("{:?}", window.layout_bounds(id))).collect();
    let calls = calls.borrow().clone();
    (bounds, calls)
}

#[test]
fn a_root_laid_out_less_often_than_last_frame_is_laid_out_anew() {
    let model = Rc::new(RefCell::new((
        Node::Text { words: vec![0], nowrap: false, truncate: false, clamp: false },
        100.,
    )));
    let (mut app, handle) = window(model);
    let frames = app
        .update(|cx| {
            cx.update_window(handle, |_, window, cx| {
                let engine =
                    |window: &mut crate::Window| window.layout_engine.as_mut().expect("engine").clear();
                engine(window);
                lay_out(window, cx, &[60., 30.]);
                engine(window);
                // The root receives the first of last frame's two requests only: computed again, from scratch.
                let reused = lay_out(window, cx, &[60.]);
                engine(window);
                engine(window);
                let fresh = lay_out(window, cx, &[60.]);
                (reused, fresh)
            })
        })
        .expect("window");
    assert_eq!(frames.0, frames.1);
}
