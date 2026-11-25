//! The window's layout tree: taffy's algorithms (flexbox, grid, block, leaf, rounding) run over our own
//! node storage, which keeps what the previous frame computed so an unchanged subtree is not laid out
//! again. Owner tag: `"window"`.
//!
//! Each frame the elements add their nodes again (children before parents). A new node whose style, kind,
//! measure key and children are those of a node of the previous frame (its children reused from that
//! node's children) takes that node's place: its subtree is the same, so it is *reusable*. A reusable node
//! keeps the list of every layout request it received last frame with its answer (its records). This frame,
//! while the requests it receives are the same requests in the same order, it answers from the list and
//! nothing under it runs; the measured nodes under it then replay last frame's measure calls in order, so
//! their state (the text element keeps its shaped lines there) is what a fresh layout leaves.
//!
//! The first request that differs makes the node *diverge*: its subtree forgets everything (records, taffy
//! caches, measured state), the requests it already answered are computed again from scratch, and from
//! then on it is laid out as a new node. A node that received only the start of last frame's list is
//! recomputed the same way when the layout ends. So every result, every measure call a text element sees
//! and every final position is exactly what laying out a fresh tree gives (GPUI's behaviour), and
//! `TaffyTree`'s cost is paid only for what changed. The request handling is `TaffyTree`'s
//! (`compute_child_layout`: hidden layout, the per-frame cache, dispatch by display and children).
//!
//! What a record depends on, and how each is covered: the node's taffy style (compared; it is in device
//! pixels, so the rem size and the scale applied to lengths are in it), its kind and measure key (compared;
//! the text element's key holds its text, resolved font size and line height, fonts and wrapping style),
//! its children (compared, through the nodes they were reused from), the requests (compared one by one),
//! the scale factor the measure function's logical results were converted with (a frame at another
//! scale reuses nothing), and the text system's font generation (after fonts are added, a family can
//! resolve to another face: a node added under another generation is not reused). Nodes measured
//! without a key are never reused. Text shaping comes from the window's line layout cache, which a
//! replay keeps warm, so a measure in a reused subtree sees what a fresh layout of this window would.
//!
//! taffy's per-frame cache has one exception: a node's cached final layout does not answer once the
//! node has been measured again (computed in `ComputeSize`) after it was stored. taffy's block
//! algorithm lays a block's children out while it measures it, at the measuring size, so that answer
//! would leave the children where the measure put them (the cards of FOLO Desktop's run page kept
//! their content at their min-content width, beside a log of long wrapped lines).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use smallvec::SmallVec;
use taffy::{
    AvailableSpace as TaffySpace, Cache, Display, Layout, LayoutInput, LayoutOutput, NodeId, RunMode, Size,
    Style, compute_block_layout, compute_cached_layout, compute_flexbox_layout, compute_grid_layout,
    compute_hidden_layout, compute_leaf_layout, compute_root_layout, round_layout,
};

use crate::{App, AvailableSpace, Pixels, Window, size, text::fx_hash::FxHasher};

/// A measure function for a leaf whose size depends on its content (text, `canvas` with a size).
pub(crate) type NodeMeasureFn = Box<
    dyn FnMut(
        crate::Size<Option<Pixels>>,
        crate::Size<AvailableSpace>,
        &mut Window,
        &mut App,
    ) -> crate::Size<Pixels>,
>;

/// What a measured node's result depends on besides the requests (text content/style, or a uniform
/// list's count and row size). Nodes measured without one are never reused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MeasureKey {
    /// A text element's key: a hash of everything the measure function reads, plus the text itself
    /// (compared directly, so a hash collision cannot reuse another text's layout).
    Text { hash: u64, text: crate::SharedString },
    /// A `UniformList`'s key: exact item count and measured row dimensions (bit patterns).
    UniformList { item_count: usize, width_bits: u32, height_bits: u32 },
}

impl MeasureKey {
    fn hash_into(&self, hasher: &mut FxHasher) {
        match self {
            MeasureKey::Text { hash, .. } => {
                0u8.hash(hasher);
                hash.hash(hasher);
            }
            MeasureKey::UniformList { item_count, width_bits, height_bits } => {
                1u8.hash(hasher);
                item_count.hash(hasher);
                width_bits.hash(hasher);
                height_bits.hash(hasher);
            }
        }
    }
}

/// A measured node: its function, its key, and how to forget the state measuring left behind.
pub(crate) struct Measured {
    /// The measure function.
    pub(crate) measure: NodeMeasureFn,
    /// What its result depends on (`None`: never reused).
    pub(crate) key: Option<MeasureKey>,
    /// Clears the state measuring left (so a replay or a fresh layout starts from nothing).
    pub(crate) reset: Option<Box<dyn Fn()>>,
}

/// One measure call (taffy's device pixels), kept for the replay.
type Invocation = (Size<Option<f32>>, Size<TaffySpace>);

/// A node of the tree.
struct Node {
    style: Style,
    /// `children[first..first + count]` of the tree.
    first: u32,
    count: u32,
    parent: Option<NodeId>,
    measured: Option<Measured>,
    /// A hash of the subtree's content (style, kind, key, children's signatures).
    sig: u64,
    /// The previous frame's node this one took the place of.
    reused_from: Option<u32>,
    /// The subtree is last frame's: requests are answered from `records` while they match.
    reusable: bool,
    /// Every request and answer (last frame's while reusable, else this frame's), in order: each distinct
    /// pair once in `answers`, and `requests` indexes them (a node is asked the same thing many times).
    answers: SmallVec<[(LayoutInput, LayoutOutput); 1]>,
    requests: SmallVec<[u32; 4]>,
    /// How many of `requests` this frame's requests matched.
    matched: u32,
    /// The measure calls (last frame's while reusable and not measured again).
    invocations: SmallVec<[Invocation; 1]>,
    /// The measure function ran (or replayed) this frame.
    measured_now: bool,
    /// The text system's font generation when the node was added (reuse needs it unchanged).
    font_generation: u64,
    unrounded: Layout,
    final_layout: Layout,
}

impl Node {
    /// Appends a request and its answer to the records.
    fn record(&mut self, inputs: LayoutInput, output: LayoutOutput) {
        let k = match self.answers.iter().position(|&(i, o)| i == inputs && o == output) {
            Some(k) => k,
            None => {
                self.answers.push((inputs, output));
                self.answers.len() - 1
            }
        };
        self.requests.push(k as u32);
    }
}

/// The layout tree of one window (see the module docs).
#[derive(Default)]
pub(crate) struct LayoutTree {
    nodes: Vec<Node>,
    /// taffy's per-frame cache of each of this frame's nodes (last frame's are not kept).
    caches: Vec<Cache>,
    /// Per node: computed in `ComputeSize` since its cached final layout was stored, so that entry
    /// must not answer (`CacheTree::cache_get`).
    remeasured: Vec<bool>,
    children: Vec<NodeId>,
    /// Last frame's nodes, and their children, while this frame's are added.
    previous: Vec<Node>,
    previous_children: Vec<NodeId>,
    /// The scale factor of this frame's nodes, and of last frame's (`f32` bits; reuse needs them equal).
    scale: u32,
    previous_scale: u32,
    /// Last frame's nodes by signature.
    by_sig: HashMap<u64, SmallVec<[u32; 1]>, std::hash::BuildHasherDefault<FxHasher>>,
    /// Which of last frame's nodes were taken.
    taken: Vec<bool>,
    /// This frame's requests answered from records, and computed.
    pub(crate) answered: u32,
    pub(crate) computed: u32,
    /// The same for the last frame.
    pub(crate) answered_last: u32,
    pub(crate) computed_last: u32,
}

impl LayoutTree {
    /// Starts a frame: this frame's nodes become last frame's, and the tree is empty.
    pub(crate) fn clear(&mut self) {
        std::mem::swap(&mut self.nodes, &mut self.previous);
        std::mem::swap(&mut self.children, &mut self.previous_children);
        self.nodes.clear();
        self.caches.clear();
        self.remeasured.clear();
        self.previous_scale = self.scale;
        self.children.clear();
        // Room for this frame's nodes, about as many as last frame's: without the trim, doubling leaves
        // up to twice that in each of the two node lists (1.3 KB a node).
        let room = |len: usize| len + len / 8 + 16;
        trim(&mut self.nodes, room(self.previous.len()));
        trim(&mut self.caches, room(self.previous.len()));
        trim(&mut self.remeasured, room(self.previous.len()));
        trim(&mut self.children, room(self.previous_children.len()));
        self.by_sig.clear();
        for (ix, node) in self.previous.iter().enumerate() {
            self.by_sig.entry(node.sig).or_default().push(ix as u32);
        }
        self.taken.clear();
        self.taken.resize(self.previous.len(), false);
        self.answered_last = std::mem::take(&mut self.answered);
        self.computed_last = std::mem::take(&mut self.computed);
    }

    /// Adds a node with `children`; `measured` for a measured leaf. `scale_factor` is the frame's (the
    /// one `style` was converted with and `compute` will measure with), `font_generation` the text
    /// system's.
    pub(crate) fn add(
        &mut self,
        style: Style,
        children: &[NodeId],
        measured: Option<Measured>,
        scale_factor: f32,
        font_generation: u64,
    ) -> NodeId {
        self.scale = scale_factor.to_bits();
        let id = NodeId::from(self.nodes.len());
        let mut hasher = FxHasher::default();
        hash_style(&style, &mut hasher);
        measured.is_some().hash(&mut hasher);
        if let Some(key) = measured.as_ref().and_then(|m| m.key.as_ref()) {
            key.hash_into(&mut hasher);
        }
        for child in children {
            self.nodes[usize::from(*child)].sig.hash(&mut hasher);
        }
        let sig = hasher.finish();
        let reuse = self.find_previous(sig, &style, children, measured.as_ref(), font_generation);
        let first = self.children.len() as u32;
        self.children.extend_from_slice(children);
        for child in children {
            self.nodes[usize::from(*child)].parent = Some(id);
        }
        let mut node = Node {
            style,
            first,
            count: children.len() as u32,
            parent: None,
            measured,
            sig,
            reused_from: None,
            reusable: false,
            answers: SmallVec::new(),
            requests: SmallVec::new(),
            matched: 0,
            invocations: SmallVec::new(),
            measured_now: false,
            font_generation,
            unrounded: Layout::new(),
            final_layout: Layout::new(),
        };
        if let Some(prev) = reuse {
            self.taken[prev as usize] = true;
            let old = &mut self.previous[prev as usize];
            node.reused_from = Some(prev);
            node.reusable = true;
            node.answers = std::mem::take(&mut old.answers);
            node.requests = std::mem::take(&mut old.requests);
            node.invocations = std::mem::take(&mut old.invocations);
            node.unrounded = old.unrounded;
        }
        self.nodes.push(node);
        self.caches.push(Cache::new());
        self.remeasured.push(false);
        id
    }

    /// Last frame's node with this content whose children are the ones `children` were reused from.
    fn find_previous(
        &self,
        sig: u64,
        style: &Style,
        children: &[NodeId],
        measured: Option<&Measured>,
        font_generation: u64,
    ) -> Option<u32> {
        if self.scale != self.previous_scale {
            return None;
        }
        let key = match measured {
            Some(Measured { key: None, .. }) => return None,
            Some(Measured { key: Some(key), .. }) => Some(key),
            None => None,
        };
        if children.iter().any(|child| {
            let child = &self.nodes[usize::from(*child)];
            !child.reusable || child.reused_from.is_none()
        }) {
            return None;
        }
        self.by_sig.get(&sig)?.iter().copied().find(|&ix| {
            let old = &self.previous[ix as usize];
            !self.taken[ix as usize]
                && old.font_generation == font_generation
                && old.count as usize == children.len()
                && old.measured.as_ref().map(|m| m.key.as_ref()) == measured.map(|_| key)
                && old.style == *style
                && self.previous_children[old.first as usize..(old.first + old.count) as usize]
                    .iter()
                    .zip(children)
                    .all(|(&prev_child, &child)| {
                        self.nodes[usize::from(child)].reused_from == Some(usize::from(prev_child) as u32)
                    })
        })
    }

    /// Lays out the tree under `root` in `available` (taffy's `compute_root_layout` and rounding), measuring
    /// with the window; returns nothing, the layouts are read with `final_layout`.
    pub(crate) fn compute(
        &mut self,
        root: NodeId,
        available: Size<TaffySpace>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let scale_factor = window.scale_factor();
        let mut view = View { tree: self, window, cx, scale_factor };
        compute_root_layout(&mut view, root, available);
        view.finish(root);
        round_layout(&mut view, root);
    }

    /// Bytes the tree holds on the heap (both frames' nodes, children lists and records).
    pub(crate) fn heap_bytes(&self) -> usize {
        let records = |nodes: &[Node]| -> usize {
            nodes
                .iter()
                .map(|n| {
                    let answers = if n.answers.spilled() {
                        n.answers.capacity() * std::mem::size_of::<(LayoutInput, LayoutOutput)>()
                    } else {
                        0
                    };
                    let requests = if n.requests.spilled() {
                        n.requests.capacity() * std::mem::size_of::<u32>()
                    } else {
                        0
                    };
                    let invocations = if n.invocations.spilled() {
                        n.invocations.capacity() * std::mem::size_of::<Invocation>()
                    } else {
                        0
                    };
                    answers + requests + invocations
                })
                .sum()
        };
        (self.nodes.capacity() + self.previous.capacity()) * std::mem::size_of::<Node>()
            + self.caches.capacity() * std::mem::size_of::<Cache>()
            + self.remeasured.capacity()
            + (self.children.capacity() + self.previous_children.capacity()) * std::mem::size_of::<NodeId>()
            + records(&self.nodes)
            + records(&self.previous)
    }

    /// A node's rounded layout (relative to its parent).
    pub(crate) fn final_layout(&self, id: NodeId) -> &Layout {
        &self.nodes[usize::from(id)].final_layout
    }

    /// A node's parent.
    pub(crate) fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes[usize::from(id)].parent
    }

    /// A node's children.
    pub(crate) fn children(&self, id: NodeId) -> &[NodeId] {
        let node = &self.nodes[usize::from(id)];
        &self.children[node.first as usize..(node.first + node.count) as usize]
    }
}

#[cfg(test)]
impl LayoutTree {
    /// One layout request to `node`, as a parent's algorithm makes it (tests replay a sequence).
    pub(crate) fn request(
        &mut self,
        node: NodeId,
        inputs: LayoutInput,
        window: &mut Window,
        cx: &mut App,
    ) -> LayoutOutput {
        use taffy::LayoutPartialTree as _;
        let scale_factor = window.scale_factor();
        View { tree: self, window, cx, scale_factor }.compute_child_layout(node, inputs)
    }

    /// A node's unrounded layout (relative to its parent), as the last request left it.
    pub(crate) fn unrounded(&self, id: NodeId) -> &Layout {
        &self.nodes[usize::from(id)].unrounded
    }
}

/// Gives back `v`'s capacity beyond `room` (and reserves up to it), keeping steady frames free of
/// reallocation.
fn trim<T>(v: &mut Vec<T>, room: usize) {
    if v.capacity() > room + room / 2 {
        v.shrink_to(room);
    }
    v.reserve_exact(room.saturating_sub(v.len()));
}

/// The tree with the window, while laying out (what taffy's algorithms run over).
struct View<'a> {
    tree: &'a mut LayoutTree,
    window: &'a mut Window,
    cx: &'a mut App,
    scale_factor: f32,
}

impl View<'_> {
    /// `TaffyTree::compute_child_layout`, recorded.
    fn compute_fresh(&mut self, node: NodeId, inputs: LayoutInput) -> LayoutOutput {
        self.tree.computed += 1;
        let output = if inputs.run_mode == RunMode::PerformHiddenLayout {
            compute_hidden_layout(self, node)
        } else {
            compute_cached_layout(self, node, inputs, |view, node, inputs| {
                let ix = usize::from(node);
                // Measuring lays the children out too (taffy's block algorithm runs their final
                // layout at the size it measures with), so a final layout cached before no longer
                // describes them.
                if inputs.run_mode == RunMode::ComputeSize {
                    view.tree.remeasured[ix] = true;
                }
                let display = view.tree.nodes[ix].style.display;
                match (display, view.tree.nodes[ix].count > 0) {
                    (Display::None, _) => compute_hidden_layout(view, node),
                    (Display::Block, true) => compute_block_layout(view, node, inputs),
                    (Display::Flex, true) => compute_flexbox_layout(view, node, inputs),
                    (Display::Grid, true) => compute_grid_layout(view, node, inputs),
                    (_, false) => view.compute_leaf(ix, inputs),
                }
            })
        };
        self.tree.nodes[usize::from(node)].record(inputs, output);
        output
    }

    /// A leaf: taffy's leaf layout, measuring through the node's function (in logical pixels, as GPUI).
    fn compute_leaf(&mut self, ix: usize, inputs: LayoutInput) -> LayoutOutput {
        let (scale_factor, window, cx) = (self.scale_factor, &mut *self.window, &mut *self.cx);
        let node = &mut self.tree.nodes[ix];
        let (style, measured, invocations) = (&node.style, &mut node.measured, &mut node.invocations);
        let measured_now = &mut node.measured_now;
        compute_leaf_layout(
            inputs,
            style,
            |_, _| 0.0,
            |known, available| {
                let Some(measured) = measured.as_mut() else { return Size::ZERO };
                invocations.push((known, available));
                *measured_now = true;
                measure(&mut measured.measure, known, available, scale_factor, window, cx)
            },
        )
    }

    /// Forgets everything under `ix` (it is laid out as new from now on); `unplaced` also gives the
    /// subtree a new node's layout.
    fn reset_subtree(&mut self, ix: usize, unplaced: bool) {
        let mut stack: SmallVec<[usize; 32]> = SmallVec::new();
        stack.push(ix);
        while let Some(ix) = stack.pop() {
            let node = &mut self.tree.nodes[ix];
            node.reusable = false;
            node.answers.clear();
            node.requests.clear();
            node.matched = 0;
            node.invocations.clear();
            self.tree.caches[ix].clear();
            self.tree.remeasured[ix] = false;
            if unplaced {
                node.unrounded = Layout::new();
            }
            if node.measured_now {
                node.measured_now = false;
                if let Some(reset) = node.measured.as_ref().and_then(|m| m.reset.as_ref()) {
                    reset();
                }
            }
            let (first, count) = (node.first as usize, node.count as usize);
            stack.extend(self.tree.children[first..first + count].iter().map(|c| usize::from(*c)));
        }
    }

    /// `ix` stops being reusable: its subtree is forgotten and the requests it answered are computed again.
    fn diverge(&mut self, ix: usize) {
        let node = &self.tree.nodes[ix];
        let answered: SmallVec<[LayoutInput; 8]> =
            node.requests[..node.matched as usize].iter().map(|&k| node.answers[k as usize].0).collect();
        self.reset_subtree(ix, false);
        for inputs in answered {
            self.compute_fresh(NodeId::from(ix), inputs);
        }
    }

    /// After the layout: a reused node that received only the start of last frame's requests is computed
    /// again from scratch, one that no request reached (its parent laid out anew, without it) is reset to
    /// what a new node holds, and measured nodes laid out from records replay last frame's measure calls.
    fn finish(&mut self, root: NodeId) {
        // (node, whether an ancestor answered all its requests from records)
        let mut stack: SmallVec<[(usize, bool); 64]> = SmallVec::new();
        stack.push((usize::from(root), false));
        while let Some((ix, covered)) = stack.pop() {
            let node = &self.tree.nodes[ix];
            let mut covers = false;
            if node.reusable {
                let (matched, len) = (node.matched as usize, node.requests.len());
                if matched > 0 && matched < len {
                    self.diverge(ix);
                    continue;
                }
                if matched == 0 && len > 0 && !covered {
                    self.reset_subtree(ix, true);
                    continue;
                }
                if !node.measured_now && !node.invocations.is_empty() {
                    self.replay(ix);
                }
                covers = covered || len > 0;
            }
            let node = &self.tree.nodes[ix];
            let (first, count) = (node.first as usize, node.count as usize);
            stack.extend(self.tree.children[first..first + count].iter().map(|c| (usize::from(*c), covers)));
        }
    }

    /// Runs a reusable measured node's recorded calls on this frame's function, in order.
    fn replay(&mut self, ix: usize) {
        let (scale_factor, window, cx) = (self.scale_factor, &mut *self.window, &mut *self.cx);
        let node = &mut self.tree.nodes[ix];
        node.measured_now = true;
        if let Some(measured) = node.measured.as_mut() {
            for &(known, available) in &node.invocations {
                measure(&mut measured.measure, known, available, scale_factor, window, cx);
            }
        }
    }
}

/// Calls a measure function in logical pixels (taffy works in device pixels): GPUI's unit conversion.
fn measure(
    measure: &mut NodeMeasureFn,
    known: Size<Option<f32>>,
    available: Size<TaffySpace>,
    scale_factor: f32,
    window: &mut Window,
    cx: &mut App,
) -> Size<f32> {
    let known = crate::Size {
        width: known.width.map(|e| Pixels(e / scale_factor)),
        height: known.height.map(|e| Pixels(e / scale_factor)),
    };
    let unscale = |space: TaffySpace| match AvailableSpace::from(space) {
        AvailableSpace::Definite(pixels) => AvailableSpace::Definite(Pixels(pixels.0 / scale_factor)),
        other => other,
    };
    let available = size(unscale(available.width), unscale(available.height));
    let measured = measure(known, available, window, cx);
    Size { width: measured.width.0 * scale_factor, height: measured.height.0 * scale_factor }
}

/// Hashes the style fields `Style::to_taffy` sets (the rest are taffy's defaults); equality is checked
/// on the whole style before a node is reused.
fn hash_style(style: &Style, h: &mut FxHasher) {
    use taffy::{CompactLength, Dimension, LengthPercentage, LengthPercentageAuto};
    let mut raw = |v: CompactLength| {
        h.write_usize(v.tag());
        h.write_u32(v.value().to_bits());
    };
    let lpa = |v: LengthPercentageAuto| v.into_raw();
    let lp = |v: LengthPercentage| v.into_raw();
    let dim = |v: Dimension| v.into_raw();
    for v in [style.size.width, style.size.height, style.min_size.width, style.min_size.height] {
        raw(dim(v));
    }
    for v in [style.max_size.width, style.max_size.height, style.flex_basis] {
        raw(dim(v));
    }
    for r in [style.inset, style.margin] {
        for v in [r.left, r.right, r.top, r.bottom] {
            raw(lpa(v));
        }
    }
    for r in [style.padding, style.border] {
        for v in [r.left, r.right, r.top, r.bottom] {
            raw(lp(v));
        }
    }
    raw(lp(style.gap.width));
    raw(lp(style.gap.height));
    h.write_u8(style.display as u8);
    h.write_u8(style.position as u8);
    h.write_u8(style.overflow.x as u8);
    h.write_u8(style.overflow.y as u8);
    h.write_u8(style.flex_direction as u8);
    h.write_u8(style.flex_wrap as u8);
    h.write_u8(style.align_items.map_or(255, |v| v as u8));
    h.write_u8(style.align_self.map_or(255, |v| v as u8));
    h.write_u8(style.align_content.map_or(255, |v| v as u8));
    h.write_u8(style.justify_content.map_or(255, |v| v as u8));
    h.write_u32(style.flex_grow.to_bits());
    h.write_u32(style.flex_shrink.to_bits());
    h.write_u32(style.scrollbar_width.to_bits());
    h.write_u32(style.aspect_ratio.map_or(u32::MAX, f32::to_bits));
    h.write_usize(style.grid_template_rows.len());
    h.write_usize(style.grid_template_columns.len());
}

impl taffy::TraversePartialTree for View<'_> {
    type ChildIter<'a>
        = std::iter::Copied<std::slice::Iter<'a, NodeId>>
    where
        Self: 'a;

    fn child_ids(&self, parent: NodeId) -> Self::ChildIter<'_> {
        self.tree.children(parent).iter().copied()
    }

    fn child_count(&self, parent: NodeId) -> usize {
        self.tree.nodes[usize::from(parent)].count as usize
    }

    fn get_child_id(&self, parent: NodeId, index: usize) -> NodeId {
        self.tree.children(parent)[index]
    }
}

impl taffy::TraverseTree for View<'_> {}

impl taffy::LayoutPartialTree for View<'_> {
    type CoreContainerStyle<'a>
        = &'a Style
    where
        Self: 'a;

    type CustomIdent = String;

    fn get_core_container_style(&self, node: NodeId) -> Self::CoreContainerStyle<'_> {
        &self.tree.nodes[usize::from(node)].style
    }

    fn set_unrounded_layout(&mut self, node: NodeId, layout: &Layout) {
        self.tree.nodes[usize::from(node)].unrounded = *layout;
    }

    fn resolve_calc_value(&self, _val: *const (), _basis: f32) -> f32 {
        0.0
    }

    fn compute_child_layout(&mut self, node: NodeId, inputs: LayoutInput) -> LayoutOutput {
        let ix = usize::from(node);
        let n = &mut self.tree.nodes[ix];
        if n.reusable {
            if let Some(&k) = n.requests.get(n.matched as usize)
                && let (recorded, output) = n.answers[k as usize]
                && recorded == inputs
            {
                n.matched += 1;
                self.tree.answered += 1;
                return output;
            }
            self.diverge(ix);
        }
        self.compute_fresh(node, inputs)
    }
}

impl taffy::CacheTree for View<'_> {
    fn cache_get(
        &self,
        node: NodeId,
        known_dimensions: Size<Option<f32>>,
        available_space: Size<TaffySpace>,
        run_mode: RunMode,
    ) -> Option<LayoutOutput> {
        let ix = usize::from(node);
        // A final layout cached before the node was measured again: measuring moved its children
        // (taffy's block algorithm lays them out at the measuring size), and an answer from this
        // entry would leave them there. A card laid out at its shrink-to-fit width while measured
        // kept its content at that width once stretched (the run page's charts, beside a long log).
        if run_mode == RunMode::PerformLayout && self.tree.remeasured[ix] {
            return None;
        }
        self.tree.caches[ix].get(known_dimensions, available_space, run_mode)
    }

    fn cache_store(
        &mut self,
        node: NodeId,
        known_dimensions: Size<Option<f32>>,
        available_space: Size<TaffySpace>,
        run_mode: RunMode,
        output: LayoutOutput,
    ) {
        let ix = usize::from(node);
        if run_mode == RunMode::PerformLayout {
            self.tree.remeasured[ix] = false;
        }
        self.tree.caches[ix].store(known_dimensions, available_space, run_mode, output)
    }

    fn cache_clear(&mut self, node: NodeId) {
        let ix = usize::from(node);
        self.tree.caches[ix].clear();
        self.tree.remeasured[ix] = false;
    }
}

impl taffy::LayoutBlockContainer for View<'_> {
    type BlockContainerStyle<'a>
        = &'a Style
    where
        Self: 'a;
    type BlockItemStyle<'a>
        = &'a Style
    where
        Self: 'a;

    fn get_block_container_style(&self, node: NodeId) -> Self::BlockContainerStyle<'_> {
        &self.tree.nodes[usize::from(node)].style
    }

    fn get_block_child_style(&self, child: NodeId) -> Self::BlockItemStyle<'_> {
        &self.tree.nodes[usize::from(child)].style
    }
}

impl taffy::LayoutFlexboxContainer for View<'_> {
    type FlexboxContainerStyle<'a>
        = &'a Style
    where
        Self: 'a;
    type FlexboxItemStyle<'a>
        = &'a Style
    where
        Self: 'a;

    fn get_flexbox_container_style(&self, node: NodeId) -> Self::FlexboxContainerStyle<'_> {
        &self.tree.nodes[usize::from(node)].style
    }

    fn get_flexbox_child_style(&self, child: NodeId) -> Self::FlexboxItemStyle<'_> {
        &self.tree.nodes[usize::from(child)].style
    }
}

impl taffy::LayoutGridContainer for View<'_> {
    type GridContainerStyle<'a>
        = &'a Style
    where
        Self: 'a;
    type GridItemStyle<'a>
        = &'a Style
    where
        Self: 'a;

    fn get_grid_container_style(&self, node: NodeId) -> Self::GridContainerStyle<'_> {
        &self.tree.nodes[usize::from(node)].style
    }

    fn get_grid_child_style(&self, child: NodeId) -> Self::GridItemStyle<'_> {
        &self.tree.nodes[usize::from(child)].style
    }
}

impl taffy::RoundTree for View<'_> {
    fn get_unrounded_layout(&self, node: NodeId) -> Layout {
        self.tree.nodes[usize::from(node)].unrounded
    }

    fn set_final_layout(&mut self, node: NodeId, layout: &Layout) {
        self.tree.nodes[usize::from(node)].final_layout = *layout;
    }
}
