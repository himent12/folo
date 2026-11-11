// Adapted from GPUI (Apache-2.0): crates/gpui/src/util.rs (FluentBuilder)
//! `FluentBuilder`: conditional steps in a builder chain. GPUI implements it for every `IntoElement`
//! (the element layer adds that blanket impl).

/// Conditional and mapped steps for builder chains (`map`, `when`, `when_some`, …).
pub trait FluentBuilder {
    /// Passes `self` through `f`.
    fn map<U>(self, f: impl FnOnce(Self) -> U) -> U
    where
        Self: Sized,
    {
        f(self)
    }

    /// Applies `then` when `condition` holds.
    fn when(self, condition: bool, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if condition { then(this) } else { this })
    }

    /// Applies `then` when `condition` holds, else `else_fn`.
    fn when_else(
        self,
        condition: bool,
        then: impl FnOnce(Self) -> Self,
        else_fn: impl FnOnce(Self) -> Self,
    ) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if condition { then(this) } else { else_fn(this) })
    }

    /// Applies `then` with the value when `option` is `Some`.
    fn when_some<T>(self, option: Option<T>, then: impl FnOnce(Self, T) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if let Some(value) = option { then(this, value) } else { this })
    }

    /// Applies `then` when `option` is `None`.
    fn when_none<T>(self, option: &Option<T>, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if option.is_some() { this } else { then(this) })
    }
}
