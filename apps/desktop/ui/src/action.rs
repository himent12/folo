//! Actions: the `Action` trait and the `actions!` macro. Owner tag: `"runtime"`.
//!
//! The app's actions are all unit structs declared with `actions!(namespace, [A, B, ..])`; there is no JSON
//! action registry (nothing builds actions by name).

use std::any::Any;

/// Something a key binding or `Window::dispatch_action` sends through the dispatch tree.
pub trait Action: Any + Send {
    /// A boxed copy.
    fn boxed_clone(&self) -> Box<dyn Action>;

    /// Whether `action` is equal to this one (same type, equal value).
    fn partial_eq(&self, action: &dyn Action) -> bool;

    /// `"namespace::Name"`.
    fn name(&self) -> &'static str;

    /// The action as `Any`, to downcast in listeners.
    fn as_any(&self) -> &dyn Any;
}

impl std::fmt::Debug for dyn Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Declares unit-struct actions: `actions!(editor, [Copy, Paste])` makes `pub struct Copy;` named
/// `"editor::Copy"`, and so on.
#[macro_export]
macro_rules! actions {
    ($namespace:path, [ $( $(#[$attr:meta])* $name:ident ),* $(,)? ]) => {
        $(
            $(#[$attr])*
            #[derive(Clone, PartialEq, Default, Debug)]
            pub struct $name;

            impl $crate::Action for $name {
                fn boxed_clone(&self) -> ::std::boxed::Box<dyn $crate::Action> {
                    ::std::boxed::Box::new(self.clone())
                }

                fn partial_eq(&self, action: &dyn $crate::Action) -> bool {
                    action.as_any().downcast_ref::<Self>().is_some_and(|a| a == self)
                }

                fn name(&self) -> &'static str {
                    concat!(stringify!($namespace), "::", stringify!($name))
                }

                fn as_any(&self) -> &dyn ::std::any::Any {
                    self
                }
            }
        )*
    };
}

#[cfg(test)]
mod tests {
    use super::Action;

    crate::actions!(test_ns, [Alpha, Beta]);

    #[test]
    fn actions_macro_names_and_compares() {
        assert_eq!(Alpha.name(), "test_ns::Alpha");
        assert!(Alpha.partial_eq(&Alpha));
        assert!(!Alpha.partial_eq(&Beta));
        assert!(Alpha.boxed_clone().partial_eq(&Alpha));
    }
}
