//! Containers: [`Sequential`] and [`ModuleList`].

use folo_core::Tensor;

use crate::Result;
use crate::module::{Forward, Layer, Module};

/// Layers applied in order. Children are named by position (`0`, `1`, ...) unless pushed with a name.
#[derive(Default)]
pub struct Sequential {
    layers: Vec<(String, Box<dyn Layer>)>,
}

impl Sequential {
    /// An empty sequence (forward is the identity).
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a layer named by its position.
    pub fn push(mut self, layer: impl Layer) -> Self {
        let name = self.layers.len().to_string();
        self.layers.push((name, Box::new(layer)));
        self
    }

    /// Append a layer with an explicit name.
    pub fn push_named(mut self, name: impl Into<String>, layer: impl Layer) -> Self {
        self.layers.push((name.into(), Box::new(layer)));
        self
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// True if there are no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The layer at `i`.
    pub fn get(&self, i: usize) -> Option<&dyn Layer> {
        self.layers.get(i).map(|(_, l)| l.as_ref())
    }
}

impl Module for Sequential {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        for (name, l) in &self.layers {
            f(name, l.as_ref());
        }
    }
}

impl Forward for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.clone();
        for (_, l) in &self.layers {
            y = l.forward(&y)?;
        }
        Ok(y)
    }
}

/// A list of modules of one type, named `<prefix><index>` (`block0`, `block1`, ... with prefix `block`;
/// PyTorch-style `0`, `1`, ... with an empty prefix).
///
/// A parent that wants the items directly under its own prefix (docs/MODEL.md: `<p>.block0.*`, not
/// `<p>.blocks.block0.*`) forwards the list's children from its own `visit_children`:
///
/// ```
/// use folo_core::rng::Rng;
/// use folo_nn::{Conv2d, Conv2dConfig, Module, ModuleList};
///
/// struct Stage { blocks: ModuleList<Conv2d> }
/// impl Module for Stage {
///     fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
///         self.blocks.visit_children(f);
///     }
/// }
/// let mut rng = Rng::seed(0);
/// let blocks = (0..2).map(|_| Conv2d::new(Conv2dConfig::new(4, 4, 1).bias(false), &mut rng)).collect();
/// let s = Stage { blocks: ModuleList::with_prefix("block", blocks) };
/// let names: Vec<_> = s.named_parameters().into_iter().map(|(n, _)| n).collect();
/// assert_eq!(names, ["block0.weight", "block1.weight"]);
/// ```
pub struct ModuleList<M> {
    prefix: String,
    items: Vec<M>,
}

impl<M> ModuleList<M> {
    /// Items named `0`, `1`, ...
    pub fn new(items: Vec<M>) -> Self {
        Self::with_prefix("", items)
    }

    /// Items named `<prefix>0`, `<prefix>1`, ...
    pub fn with_prefix(prefix: impl Into<String>, items: Vec<M>) -> Self {
        ModuleList { prefix: prefix.into(), items }
    }

    /// The items.
    pub fn items(&self) -> &[M] {
        &self.items
    }

    /// Mutable access to the items (to replace a layer, e.g. after fusion).
    pub fn items_mut(&mut self) -> &mut [M] {
        &mut self.items
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over the items.
    pub fn iter(&self) -> std::slice::Iter<'_, M> {
        self.items.iter()
    }
}

impl<M> std::ops::Index<usize> for ModuleList<M> {
    type Output = M;
    fn index(&self, i: usize) -> &M {
        &self.items[i]
    }
}

impl<M: Module> Module for ModuleList<M> {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        for (i, m) in self.items.iter().enumerate() {
            f(&format!("{}{i}", self.prefix), m);
        }
    }
}

impl<M: Layer> Forward for ModuleList<M> {
    /// Apply the items in sequence.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.clone();
        for m in &self.items {
            y = m.forward(&y)?;
        }
        Ok(y)
    }
}
