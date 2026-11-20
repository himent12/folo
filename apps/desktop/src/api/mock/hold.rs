//! Calls a test holds back: answered from what the demo held when the call was made, then kept until the
//! test releases them, so an answer can land after later calls and local changes (a slow server, a slow
//! device). The test knows when the answer is made ([`Held::wait_held`]): nothing depends on a timeout.

use std::sync::{Condvar, Mutex};

use super::MockBackend;

/// Where a held call is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    /// The next such call is held.
    Armed,
    /// One made its answer and waits.
    Held,
    /// Only tests release.
    #[cfg_attr(not(test), allow(dead_code))]
    Released,
}

/// One held call's state, shared by the test and the call.
pub(super) struct Gate {
    stage: Mutex<Stage>,
    moved: Condvar,
}

/// The test's hand on a held call ([`MockBackend::hold`]). Dropping it releases the call.
#[cfg(test)]
pub struct Held(std::sync::Arc<Gate>);

#[cfg(test)]
impl Held {
    /// Blocks until the call was made and its answer is waiting. A call that never comes fails the test.
    pub fn wait_held(&self) {
        use std::time::Duration;
        let stage = self.0.stage.lock().unwrap();
        let (stage, timeout) =
            self.0.moved.wait_timeout_while(stage, Duration::from_secs(10), |s| *s == Stage::Armed).unwrap();
        assert!(!timeout.timed_out() || *stage != Stage::Armed, "the held call was never made");
    }

    /// Whether the call's answer is still waiting.
    pub fn is_held(&self) -> bool {
        *self.0.stage.lock().unwrap() == Stage::Held
    }

    /// Lets the answer go.
    pub fn release(&self) {
        *self.0.stage.lock().unwrap() = Stage::Released;
        self.0.moved.notify_all();
    }
}

#[cfg(test)]
impl Drop for Held {
    fn drop(&mut self) {
        self.release();
    }
}

impl MockBackend {
    /// Holds the next `call` (`"system"`, `"datasets"`, `"dataset"`, `"issues"`, `"models"`, `"runs"`, `"predict"`, `"images"`,
    /// `"metrics <run id>"`, `"samples <run id>"`, `"pixels <dataset id> <image id>"`) once its answer is made,
    /// until the returned hand releases it.
    #[cfg(test)]
    pub fn hold(&self, call: &'static str) -> Held {
        let gate = std::sync::Arc::new(Gate { stage: Mutex::new(Stage::Armed), moved: Condvar::new() });
        self.holds.lock().unwrap().push((call, gate.clone()));
        Held(gate)
    }

    /// Called by `call` with its answer made: waits there while a test holds it.
    pub(super) fn hold_point(&self, call: &str) {
        let gate = {
            let mut holds = self.holds.lock().unwrap();
            let Some(at) = holds.iter().position(|(c, _)| *c == call) else { return };
            holds.remove(at).1
        };
        let mut stage = gate.stage.lock().unwrap();
        if *stage == Stage::Armed {
            *stage = Stage::Held;
            gate.moved.notify_all();
        }
        let _released = gate.moved.wait_while(stage, |s| *s == Stage::Held).unwrap();
    }
}
