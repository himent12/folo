//! Globals: one value per type (`cx.set_global(Theme::new(..))`, `cx.global::<Theme>()`). Owner tag:
//! `"runtime"`. Setting or updating a global notifies its observers and redraws every window.
#![allow(dead_code)] // skeleton

use std::any::TypeId;

use crate::App;

/// A type that can be stored as a global.
pub trait Global: 'static {}

impl App {
    /// Stores `global`, replacing any previous value of its type.
    pub fn set_global<G: Global>(&mut self, global: G) {
        self.globals_by_type.insert(TypeId::of::<G>(), Box::new(global));
        self.push_effect(crate::app::Effect::NotifyGlobalObservers { global_type: TypeId::of::<G>() });
    }

    /// The global of type `G`. Panics when there is none.
    #[track_caller]
    pub fn global<G: Global>(&self) -> &G {
        self.try_global::<G>().expect("no global of this type has been set")
    }

    /// The global of type `G`, if set.
    pub fn try_global<G: Global>(&self) -> Option<&G> {
        self.globals_by_type.get(&TypeId::of::<G>()).map(|g| g.downcast_ref::<G>().expect("global type"))
    }

    /// The global of type `G`, mutably (observers are notified). Panics when there is none.
    #[track_caller]
    pub fn global_mut<G: Global>(&mut self) -> &mut G {
        self.push_effect(crate::app::Effect::NotifyGlobalObservers { global_type: TypeId::of::<G>() });
        self.globals_by_type
            .get_mut(&TypeId::of::<G>())
            .and_then(|g| g.downcast_mut::<G>())
            .expect("no global of this type has been set")
    }

    /// Whether a global of type `G` is set.
    pub fn has_global<G: Global>(&self) -> bool {
        self.globals_by_type.contains_key(&TypeId::of::<G>())
    }

    /// Updates the global of type `G` (leased out for the call, so `f` gets the app too).
    pub fn update_global<G: Global, R>(&mut self, _f: impl FnOnce(&mut G, &mut App) -> R) -> R {
        todo!("runtime")
    }
}
