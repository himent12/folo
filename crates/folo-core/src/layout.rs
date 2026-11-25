//! Strided views over storage, and the row iterator CPU kernels use to walk them.

use crate::shape::DimVec;
use crate::{Error, Result, Shape};

/// How a tensor's elements map into its storage: `element[i0, i1, ..] = storage[offset + sum(ik * strides[k])]`.
///
/// Strides are in elements. A stride of 0 on a dimension of size > 1 means the dimension is broadcast (expanded).
/// Views (transpose, narrow, broadcast, ...) only build a new `Layout`; the storage is shared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: DimVec,
    offset: usize,
}

impl Layout {
    /// Row-major layout at offset 0.
    pub fn contiguous(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let strides = shape.contiguous_strides();
        Layout { shape, strides, offset: 0 }
    }

    /// A layout from raw parts. The caller (a backend, typically) is responsible for staying inside the storage.
    pub fn new(shape: Shape, strides: DimVec, offset: usize) -> Self {
        assert_eq!(shape.rank(), strides.len(), "Layout::new: rank of shape and strides differ");
        Layout { shape, strides, offset }
    }

    /// The logical shape.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The logical dimensions.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Strides in elements, one per dimension.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Offset of element `[0, 0, ..]` in the storage, in elements.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of logical elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// True if the elements are laid out row-major without gaps (from `offset`). Size-1 dims are ignored.
    pub fn is_contiguous(&self) -> bool {
        let mut acc = 1;
        for (&d, &s) in self.shape.dims().iter().zip(&self.strides).rev() {
            if d != 1 {
                if s != acc {
                    return false;
                }
                acc *= d;
            }
        }
        true
    }

    /// `Some(start..end)` of the storage range if contiguous.
    pub fn contiguous_range(&self) -> Option<std::ops::Range<usize>> {
        self.is_contiguous().then(|| self.offset..self.offset + self.numel())
    }

    /// One past the largest storage index this layout touches (0 for an empty view), or `None` if that
    /// overflows `usize` (such a layout can never be valid).
    pub fn storage_extent(&self) -> Option<usize> {
        // Test for a zero dim rather than `numel() == 0`: the element count of a bogus shape can wrap to 0.
        if self.shape.dims().contains(&0) {
            return Some(0);
        }
        let mut end = self.offset;
        for (&d, &s) in self.shape.dims().iter().zip(&self.strides) {
            end = end.checked_add((d - 1).checked_mul(s)?)?;
        }
        end.checked_add(1)
    }

    /// True if the layout fits a storage of `len` elements.
    pub fn fits(&self, len: usize) -> bool {
        self.storage_extent().is_some_and(|e| e <= len)
    }

    /// True if no two logical elements share a storage element (a sufficient check: sorted by stride, each
    /// dimension's stride exceeds the reach of all smaller-strided dimensions). Every layout produced by tensor
    /// views passes; broadcast layouts do not.
    pub fn is_non_overlapping(&self) -> bool {
        let mut dims: crate::shape::DimVec = (0..self.rank()).filter(|&i| self.shape[i] > 1).collect();
        dims.sort_unstable_by_key(|&i| self.strides[i]);
        let mut reach = 0usize;
        for &i in &dims {
            if self.strides[i] <= reach {
                return false;
            }
            let Some(r) = (self.shape[i] - 1).checked_mul(self.strides[i]).and_then(|v| v.checked_add(reach))
            else {
                return false;
            };
            reach = r;
        }
        true
    }

    /// True if some dimension of size > 1 has stride 0 (several logical elements share storage).
    pub fn has_broadcast(&self) -> bool {
        self.shape.dims().iter().zip(&self.strides).any(|(&d, &s)| d > 1 && s == 0)
    }

    pub(crate) fn transpose(&self, a: usize, b: usize) -> Layout {
        let mut l = self.clone();
        l.shape.dims_mut().swap(a, b);
        l.strides.swap(a, b);
        l
    }

    pub(crate) fn permute(&self, perm: &[usize]) -> Layout {
        let dims: DimVec = perm.iter().map(|&p| self.shape[p]).collect();
        let strides: DimVec = perm.iter().map(|&p| self.strides[p]).collect();
        Layout { shape: dims.into(), strides, offset: self.offset }
    }

    /// Elements `start, start+step, ..` (`len` of them) along `dim`.
    pub(crate) fn slice(&self, dim: usize, start: usize, len: usize, step: usize) -> Layout {
        let mut l = self.clone();
        if len > 0 {
            l.offset += start * self.strides[dim];
        }
        l.shape.dims_mut()[dim] = len;
        l.strides[dim] *= step;
        l
    }

    pub(crate) fn broadcast_to(&self, target: &Shape, op: &'static str) -> Result<Layout> {
        if !self.shape.broadcasts_to(target) {
            return Err(Error::ShapeMismatch { op, lhs: self.shape.clone(), rhs: target.clone() });
        }
        let off = target.rank() - self.rank();
        let mut strides = DimVec::from_elem(0, target.rank());
        for i in 0..self.rank() {
            strides[off + i] = if self.shape[i] == 1 { 0 } else { self.strides[i] };
        }
        Ok(Layout { shape: target.clone(), strides, offset: self.offset })
    }

    pub(crate) fn unsqueeze(&self, dim: usize) -> Layout {
        let mut l = self.clone();
        let s = if dim < self.rank() { self.strides[dim] * self.shape[dim] } else { 1 };
        l.shape.dims_mut().insert(dim, 1);
        l.strides.insert(dim, s);
        l
    }

    pub(crate) fn squeeze(&self, dim: usize) -> Layout {
        let mut l = self.clone();
        l.shape.dims_mut().remove(dim);
        l.strides.remove(dim);
        l
    }

    /// A view with a new shape over the same elements, if possible without copying.
    pub(crate) fn reshape(&self, shape: &Shape) -> Option<Layout> {
        if self.is_contiguous() || self.numel() == 0 {
            return Some(Layout {
                shape: shape.clone(),
                strides: shape.contiguous_strides(),
                offset: self.offset,
            });
        }
        // General case (PyTorch's computeStride): each run of old dims that is contiguous among itself can be
        // regrouped into new dims of the same total size.
        let old_d = self.shape.dims();
        let old_s = &self.strides;
        let new_d = shape.dims();
        let mut new_s = DimVec::from_elem(0, new_d.len());
        let mut view_d = new_d.len() as isize - 1;
        let mut chunk_base_stride = *old_s.last().unwrap_or(&1);
        let mut tensor_numel = 1usize;
        let mut view_numel = 1usize;
        for td in (0..old_d.len()).rev() {
            tensor_numel *= old_d[td];
            if td == 0 || (old_d[td - 1] != 1 && old_s[td - 1] != tensor_numel * chunk_base_stride) {
                while view_d >= 0 && (view_numel < tensor_numel || new_d[view_d as usize] == 1) {
                    new_s[view_d as usize] = view_numel * chunk_base_stride;
                    view_numel *= new_d[view_d as usize];
                    view_d -= 1;
                }
                if view_numel != tensor_numel {
                    return None;
                }
                if td > 0 {
                    chunk_base_stride = old_s[td - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }
        if view_d != -1 {
            return None;
        }
        Some(Layout { shape: shape.clone(), strides: new_s, offset: self.offset })
    }
}

/// A plan for walking `K` strided operands of the same logical shape row by row.
///
/// Dimensions of size 1 are dropped and adjacent dimensions that are contiguous with each other in every operand
/// are merged, so most real cases end up with one or two dimensions. Each "row" is the innermost (merged)
/// dimension: `len` elements with per-operand stride `inner[k]`.
#[derive(Debug, Clone)]
pub(crate) struct RowPlan<const K: usize> {
    /// Outer dimensions (merged), outermost first.
    outer: DimVec,
    /// Per operand, stride of each outer dim.
    outer_strides: [DimVec; K],
    /// Base offsets: where the first row starts in each operand.
    pub base: [usize; K],
    /// Length of a row.
    pub len: usize,
    /// Per operand, stride along a row.
    pub inner: [usize; K],
    /// Number of rows.
    pub rows: usize,
}

impl<const K: usize> RowPlan<K> {
    pub fn new(shape: &[usize], strides: [&[usize]; K], base: [usize; K]) -> Self {
        // Collect non-trivial dims.
        let mut dims: DimVec = DimVec::new();
        let mut st: [DimVec; K] = std::array::from_fn(|_| DimVec::new());
        for (i, &d) in shape.iter().enumerate() {
            if d == 1 {
                continue;
            }
            if let Some(&last) = dims.last() {
                let mergeable = (0..K).all(|k| st[k][st[k].len() - 1] == strides[k][i] * d);
                if mergeable {
                    let n = dims.len();
                    dims[n - 1] = last * d;
                    for k in 0..K {
                        let m = st[k].len();
                        st[k][m - 1] = strides[k][i];
                    }
                    continue;
                }
            }
            dims.push(d);
            for k in 0..K {
                st[k].push(strides[k][i]);
            }
        }
        if shape.contains(&0) {
            return RowPlan {
                outer: DimVec::new(),
                outer_strides: std::array::from_fn(|_| DimVec::new()),
                base,
                len: 0,
                inner: [0; K],
                rows: 0,
            };
        }
        let (len, inner) = match dims.pop() {
            Some(d) => (d, std::array::from_fn(|k| st[k].pop().unwrap())),
            None => (1, [0; K]),
        };
        let rows = dims.iter().product();
        RowPlan { outer: dims, outer_strides: st, base, len, inner, rows }
    }

    /// The multi-index of row `row` over the outer dimensions and where it starts in each operand. Out of line
    /// and not generic over the callback of [`RowPlan::for_rows`] (once per walk, not per row), so each op that
    /// walks rows compiles only the per-row loop.
    fn row_start(&self, row: usize) -> (DimVec, [usize; K]) {
        let nd = self.outer.len();
        let mut idx: DimVec = DimVec::from_elem(0, nd);
        let mut rem = row;
        let mut offs = self.base;
        for d in (0..nd).rev() {
            let i = rem % self.outer[d];
            rem /= self.outer[d];
            idx[d] = i;
            for (k, o) in offs.iter_mut().enumerate() {
                *o += i * self.outer_strides[k][d];
            }
        }
        (idx, offs)
    }

    /// Starting element offsets for `row`.
    pub(crate) fn start_offset(&self, row: usize) -> [usize; K] {
        self.row_start(row).1
    }

    /// Calls `f(row, offsets)` for rows in `range`, where `offsets[k]` is where row `row` starts in operand `k`.
    #[inline]
    pub fn for_rows(&self, range: std::ops::Range<usize>, mut f: impl FnMut(usize, [usize; K])) {
        if range.is_empty() {
            return;
        }
        let nd = self.outer.len();
        let (mut idx, mut offs) = self.row_start(range.start);
        for row in range {
            f(row, offs);
            // Odometer increment.
            let mut d = nd;
            while d > 0 {
                d -= 1;
                idx[d] += 1;
                for (k, o) in offs.iter_mut().enumerate() {
                    *o += self.outer_strides[k][d];
                }
                if idx[d] < self.outer[d] {
                    break;
                }
                for (k, o) in offs.iter_mut().enumerate() {
                    *o -= idx[d] * self.outer_strides[k][d];
                }
                idx[d] = 0;
            }
        }
    }
}

impl RowPlan<1> {
    /// If the last outer dimension has unit stride while the inner dimension has stride `> 1`, each outer batch
    /// is a 2D transpose of `self.len × cols` (row stride `self.inner[0]`) into `cols × self.len` contiguous
    /// output. Returns `Some((batches, cols))`.
    pub(crate) fn transpose_plane(&self) -> Option<(usize, usize)> {
        let &cols = self.outer.last()?;
        let &s = self.outer_strides[0].last()?;
        (cols > 1 && s == 1 && self.len > 1 && self.inner[0] > 1).then_some((self.rows / cols, cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguity_and_views() {
        let l = Layout::contiguous([2, 3, 4]);
        assert!(l.is_contiguous());
        let t = l.transpose(1, 2);
        assert!(!t.is_contiguous());
        assert_eq!(t.strides(), &[12, 1, 4]);
        let n = l.slice(1, 1, 2, 1);
        assert_eq!(n.offset(), 4);
        assert!(!n.is_contiguous());
        assert_eq!(n.storage_extent(), Some(4 + 12 + 4 + 3 + 1));
        let huge = Layout::new(Shape::new(&[2]), smallvec::smallvec![usize::MAX], 1);
        assert_eq!(huge.storage_extent(), None);
        assert!(!huge.fits(1));
        // An element count that wraps to zero must not look empty.
        let half = 1usize << (usize::BITS / 2);
        let wraps = Layout::new(Shape::new(&[half, half]), smallvec::smallvec![1, half], 0);
        assert!(!wraps.fits(1));
        let b = Layout::contiguous([3, 1]).broadcast_to(&Shape::new(&[2, 3, 5]), "b").unwrap();
        assert_eq!(b.strides(), &[0, 1, 0]);
        assert!(b.has_broadcast());
        assert!(!b.is_non_overlapping());
        assert!(l.permute(&[2, 0, 1]).is_non_overlapping() && n.is_non_overlapping());
        assert!(l.slice(2, 0, 2, 2).is_non_overlapping());
        let bad = Layout::new(Shape::new(&[256, 256]), smallvec::smallvec![1, 1], 0);
        assert!(!bad.is_non_overlapping());
    }

    #[test]
    fn reshape_views() {
        let l = Layout::contiguous([4, 6]).transpose(0, 1); // [6,4] strides [1,6]
        assert!(l.reshape(&Shape::new(&[24])).is_none());
        let l2 = Layout::contiguous([4, 6, 5]).slice(2, 0, 2, 1); // [4,6,2] strides [30,5,1]
        let r = l2.reshape(&Shape::new(&[24, 2])).unwrap();
        assert_eq!(r.strides(), &[5, 1]);
        let r = l2.reshape(&Shape::new(&[4, 3, 2, 2])).unwrap();
        assert_eq!(r.strides(), &[30, 10, 5, 1]);
        assert!(l2.reshape(&Shape::new(&[48])).is_none());
    }

    #[test]
    fn row_plan_merges() {
        let l = Layout::contiguous([2, 3, 4]);
        let p = RowPlan::<1>::new(l.dims(), [l.strides()], [0]);
        assert_eq!((p.rows, p.len, p.inner[0]), (1, 24, 1));
        let t = l.transpose(0, 2); // [4,3,2] strides [1,4,12]
        let p = RowPlan::<1>::new(t.dims(), [t.strides()], [0]);
        let mut seen = vec![];
        p.for_rows(0..p.rows, |_, [o]| {
            for i in 0..p.len {
                seen.push(o + i * p.inner[0]);
            }
        });
        let expect: Vec<usize> =
            (0..4).flat_map(|a| (0..3).flat_map(move |b| (0..2).map(move |c| a + b * 4 + c * 12))).collect();
        assert_eq!(seen, expect);
        // Starting mid-way gives the same offsets.
        let mut mid = vec![];
        p.for_rows(5..p.rows, |_, [o]| mid.push(o));
        let mut all = vec![];
        p.for_rows(0..p.rows, |_, [o]| all.push(o));
        assert_eq!(mid, all[5..]);
    }
}
