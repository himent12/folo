//! The app's own caption: there is no non-client hit test under winit, so the window control areas the
//! app marks (`WindowControlArea`) are acted on here, with GPUI's Windows rule: the press goes to the
//! app first and only an unhandled one drags the window, toggles maximize (a double click on the drag
//! area) or arms a caption button, which acts when released over the same button.

use crate::WindowControlArea;

/// What the window should do after a left press or release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChromeAction {
    /// Nothing.
    None,
    /// Start an OS move of the window (`drag_window`).
    Drag,
    /// Maximize, or restore a maximized window.
    ToggleMaximize,
    /// Minimize.
    Minimize,
    /// Ask to close (as the system close button does).
    Close,
}

/// The caption button pressed and not yet released.
#[derive(Debug, Default)]
pub(crate) struct Chrome {
    armed: Option<WindowControlArea>,
}

impl Chrome {
    /// A left press over `area` (what the app's hit test said), `handled` by the app or not.
    pub fn press(
        &mut self,
        area: Option<WindowControlArea>,
        handled: bool,
        click_count: usize,
    ) -> ChromeAction {
        self.armed = None;
        if handled {
            return ChromeAction::None;
        }
        match area {
            Some(WindowControlArea::Drag) if click_count == 2 => ChromeAction::ToggleMaximize,
            Some(WindowControlArea::Drag) => ChromeAction::Drag,
            Some(button) => {
                self.armed = Some(button);
                ChromeAction::None
            }
            None => ChromeAction::None,
        }
    }

    /// A left release over `area`, `handled` by the app or not.
    pub fn release(&mut self, area: Option<WindowControlArea>, handled: bool) -> ChromeAction {
        let armed = self.armed.take();
        if handled || armed != area {
            return ChromeAction::None;
        }
        match armed {
            Some(WindowControlArea::Min) => ChromeAction::Minimize,
            Some(WindowControlArea::Max) => ChromeAction::ToggleMaximize,
            Some(WindowControlArea::Close) => ChromeAction::Close,
            _ => ChromeAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use WindowControlArea::*;

    #[test]
    fn drag_area_drags_and_double_click_maximizes() {
        let mut chrome = Chrome::default();
        assert_eq!(chrome.press(Some(Drag), false, 1), ChromeAction::Drag);
        assert_eq!(chrome.press(Some(Drag), false, 2), ChromeAction::ToggleMaximize);
        assert_eq!(chrome.press(Some(Drag), true, 1), ChromeAction::None, "the app handled the press");
        assert_eq!(chrome.press(None, false, 1), ChromeAction::None);
    }

    #[test]
    fn buttons_act_on_release_inside_the_same_button() {
        let mut chrome = Chrome::default();
        assert_eq!(chrome.press(Some(Close), false, 1), ChromeAction::None);
        assert_eq!(chrome.release(Some(Close), false), ChromeAction::Close);
        chrome.press(Some(Min), false, 1);
        assert_eq!(chrome.release(Some(Min), false), ChromeAction::Minimize);
        chrome.press(Some(Max), false, 1);
        assert_eq!(chrome.release(Some(Max), false), ChromeAction::ToggleMaximize);
        // Released elsewhere: nothing, and the button is disarmed.
        chrome.press(Some(Close), false, 1);
        assert_eq!(chrome.release(Some(Min), false), ChromeAction::None);
        assert_eq!(chrome.release(Some(Close), false), ChromeAction::None);
        // The app handled the press or the release.
        chrome.press(Some(Close), true, 1);
        assert_eq!(chrome.release(Some(Close), false), ChromeAction::None);
        chrome.press(Some(Close), false, 1);
        assert_eq!(chrome.release(Some(Close), true), ChromeAction::None);
    }
}
