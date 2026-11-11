// Adapted from GPUI (Apache-2.0): crates/refineable/src/refineable.rs, crates/refineable/derive_refineable
//! `Refineable`: partial updates (refinements) merged onto full values, as GPUI's derive macro builds
//! them. A refinement field is `Some` where it overrides; `refine` copies the overrides in.
//!
//! Three kinds of field, as in GPUI's derive:
//! - `plain` `T`: the refinement holds `Option<T>`; `Some` replaces the whole value (a nested struct
//!   that is not marked refineable, such as `Style::text`, is replaced, not merged).
//! - `opt` `Option<T>`: the refinement holds `Option<T>` too; `Some` sets the value.
//! - `nested` `T` with refinement `R`: merged field by field (`Point`, `Size`, `Edges`, `Corners`).

use crate::{Corners, Edges, Point, Size};
use std::fmt::Debug;

/// A type that partial updates (its `Refinement`) can be applied to.
pub trait Refineable: Clone {
    /// The partial form of this type: every field optional.
    type Refinement: Refineable<Refinement = Self::Refinement> + Default;

    /// Applies the `Some` fields of `refinement` to `self`.
    fn refine(&mut self, refinement: &Self::Refinement);

    /// `self` with `refinement` applied.
    fn refined(self, refinement: Self::Refinement) -> Self;
}

/// The type of a field in the full struct.
macro_rules! full_ty {
    (plain $ty:ty) => { $ty };
    (opt $ty:ty) => { Option<$ty> };
    (nested $ty:ty => $rty:ty) => { $ty };
}

/// The type of a field in the refinement.
macro_rules! refinement_ty {
    (plain $ty:ty) => { Option<$ty> };
    (opt $ty:ty) => { Option<$ty> };
    (nested $ty:ty => $rty:ty) => { $rty };
}

/// `full.field.refine(&refinement.field)` for one field of a full struct.
macro_rules! refine_full {
    (plain, $full:expr, $part:expr) => {
        if let Some(value) = &$part {
            $full = value.clone();
        }
    };
    (opt, $full:expr, $part:expr) => {
        if let Some(value) = &$part {
            $full = Some(value.clone());
        }
    };
    (nested, $full:expr, $part:expr) => {
        $crate::style::Refineable::refine(&mut $full, &$part)
    };
}

/// `refinement.field.refine(&other.field)` for one field of a refinement.
macro_rules! refine_part {
    (nested, $full:expr, $part:expr) => {
        $crate::style::Refineable::refine(&mut $full, &$part)
    };
    ($kind:ident, $full:expr, $part:expr) => {
        if let Some(value) = &$part {
            $full = Some(value.clone());
        }
    };
}

/// Defines a full struct and its refinement (GPUI's `#[derive(Refineable)]`).
macro_rules! refineable {
    (
        $(#[$attr:meta])*
        pub struct $name:ident / $(#[$rattr:meta])* $refinement:ident {
            $( $(#[$fattr:meta])* $kind:ident $field:ident : $ty:ty $(=> $rty:ty)? ),* $(,)?
        }
    ) => {
        $(#[$attr])*
        pub struct $name {
            $( $(#[$fattr])* pub $field: full_ty!($kind $ty $(=> $rty)?), )*
        }

        #[doc = concat!("A refinement of [`", stringify!($name), "`]: each field `Some` where it overrides.")]
        $(#[$rattr])*
        pub struct $refinement {
            $( $(#[$fattr])* pub $field: refinement_ty!($kind $ty $(=> $rty)?), )*
        }

        impl $crate::style::Refineable for $name {
            type Refinement = $refinement;

            fn refine(&mut self, refinement: &$refinement) {
                $( refine_full!($kind, self.$field, refinement.$field); )*
            }

            fn refined(mut self, refinement: $refinement) -> Self {
                $crate::style::Refineable::refine(&mut self, &refinement);
                self
            }
        }

        impl $crate::style::Refineable for $refinement {
            type Refinement = $refinement;

            fn refine(&mut self, refinement: &$refinement) {
                $( refine_part!($kind, self.$field, refinement.$field); )*
            }

            fn refined(mut self, refinement: $refinement) -> Self {
                $crate::style::Refineable::refine(&mut self, &refinement);
                self
            }
        }

        impl $refinement {
            /// Whether any field is set.
            pub fn is_some(&self) -> bool {
                false $( || self.$field.is_some() )*
            }

            /// Whether no field is set (applying it changes nothing).
            pub fn is_empty(&self) -> bool {
                !self.is_some()
            }
        }
    };
}

pub(crate) use {full_ty, refine_full, refine_part, refineable, refinement_ty};

/// Refinements of the generic geometry types: every component optional.
macro_rules! geometry_refinement {
    ($( $(#[$attr:meta])* $ty:ident => $refinement:ident { $($field:ident),+ } )*) => {$(
        $(#[$attr])*
        #[derive(Clone, Default, Debug, PartialEq)]
        pub struct $refinement<T: Clone + Debug + Default + PartialEq> {
            $( #[doc = concat!("The `", stringify!($field), "` override.")] pub $field: Option<T>, )+
        }

        impl<T: Clone + Debug + Default + PartialEq> $refinement<T> {
            /// Whether any component is set.
            pub fn is_some(&self) -> bool {
                false $( || self.$field.is_some() )+
            }

            /// Whether no component is set.
            pub fn is_empty(&self) -> bool {
                !self.is_some()
            }
        }

        impl<T: Clone + Debug + Default + PartialEq> Refineable for $ty<T> {
            type Refinement = $refinement<T>;

            fn refine(&mut self, refinement: &$refinement<T>) {
                $( if let Some(value) = &refinement.$field { self.$field = value.clone(); } )+
            }

            fn refined(mut self, refinement: $refinement<T>) -> Self {
                self.refine(&refinement);
                self
            }
        }

        impl<T: Clone + Debug + Default + PartialEq> Refineable for $refinement<T> {
            type Refinement = $refinement<T>;

            fn refine(&mut self, refinement: &$refinement<T>) {
                $( if let Some(value) = &refinement.$field { self.$field = Some(value.clone()); } )+
            }

            fn refined(mut self, refinement: $refinement<T>) -> Self {
                self.refine(&refinement);
                self
            }
        }
    )*};
}

geometry_refinement! {
    /// A refinement of [`Point`].
    Point => PointRefinement { x, y }
    /// A refinement of [`Size`].
    Size => SizeRefinement { width, height }
    /// A refinement of [`Edges`].
    Edges => EdgesRefinement { top, right, bottom, left }
    /// A refinement of [`Corners`].
    Corners => CornersRefinement { top_left, top_right, bottom_right, bottom_left }
}
