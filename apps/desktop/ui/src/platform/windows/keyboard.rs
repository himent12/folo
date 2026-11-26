// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/events.rs (keys, IME), keyboard.rs
//! Keys, text and IME with GPUI's semantics (`events.rs`, `keyboard.rs` there): `WM_KEYDOWN` becomes a
//! `KeyDown` with GPUI's key names and `key_char`; only a key the app did not handle is translated
//! into `WM_CHAR` (so a handled shortcut never also types), and `WM_CHAR` / dead keys / IME
//! composition drive the window's `PlatformInputHandler`.

use super::util::*;
use super::window::{WindowInner, WindowState};
use crate::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::UI::Input::Ime::*;
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;

impl WindowInner {
    pub(crate) fn handle_syskeydown(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let input = {
            let mut state = self.state.borrow_mut();
            key_event(hwnd, wparam, lparam, &mut state, |keystroke| {
                PlatformInput::KeyDown(KeyDownEvent { keystroke, is_held: lparam.0 & (1 << 30) != 0 })
            })?
        };
        let result = self.dispatch_input(input)?;
        if !result.propagate {
            self.state.borrow_mut().system_key_handled = true;
            Some(0)
        } else {
            // DefWindowProc keeps the system shortcuts (Alt+F4, Alt+Space…).
            None
        }
    }

    pub(crate) fn handle_syskeyup(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let input = {
            let mut state = self.state.borrow_mut();
            key_event(hwnd, wparam, lparam, &mut state, |keystroke| {
                PlatformInput::KeyUp(KeyUpEvent { keystroke })
            })?
        };
        self.dispatch_input(input)?;
        // Always handled, so ModifiersChanged keeps working after Alt.
        Some(0)
    }

    pub(crate) fn handle_keydown(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let input = {
            let mut state = self.state.borrow_mut();
            key_event(hwnd, wparam, lparam, &mut state, |keystroke| {
                PlatformInput::KeyDown(KeyDownEvent { keystroke, is_held: lparam.0 & (1 << 30) != 0 })
            })
        };
        let Some(input) = input else { return Some(1) };
        let composing = self.with_input_handler(|h| h.marked_text_range()).flatten().is_some();
        if composing {
            translate_message(hwnd, wparam, lparam);
            return Some(0);
        }
        match self.dispatch_input(input) {
            None => Some(1),
            Some(r) if !r.propagate => Some(0),
            Some(_) => {
                translate_message(hwnd, wparam, lparam);
                Some(1)
            }
        }
    }

    pub(crate) fn handle_keyup(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let input = {
            let mut state = self.state.borrow_mut();
            key_event(hwnd, wparam, lparam, &mut state, |keystroke| {
                PlatformInput::KeyUp(KeyUpEvent { keystroke })
            })
        };
        let Some(input) = input else { return Some(1) };
        match self.dispatch_input(input) {
            Some(r) if !r.propagate => Some(0),
            _ => Some(1),
        }
    }

    pub(crate) fn handle_char(&self, wparam: WPARAM) -> Option<isize> {
        let text = parse_char(&mut self.state.borrow_mut(), wparam)?;
        self.with_input_handler(|h| h.replace_text_in_range(None, &text));
        Some(0)
    }

    pub(crate) fn handle_dead_char(&self, wparam: WPARAM) -> Option<isize> {
        let ch = char::from_u32(wparam.0 as u32)?.to_string();
        self.with_input_handler(|h| h.replace_and_mark_text_in_range(None, &ch, None));
        None
    }

    pub(crate) fn handle_system_command(&self, wparam: WPARAM) -> Option<isize> {
        if let Some(handled) = self.offscreen_system_command(wparam.0 as u32) {
            return Some(handled);
        }
        if wparam.0 == SC_KEYMENU as usize && std::mem::take(&mut self.state.borrow_mut().system_key_handled)
        {
            return Some(0);
        }
        None
    }

    /// The caret in device pixels: the app's `update_ime_position` bounds if it gave some, else the
    /// selection's bounds from the input handler (GPUI's), at the vertical middle of the line.
    fn caret_position(&self) -> Option<POINT> {
        let (given, scale) = {
            let state = self.state.borrow();
            (state.ime_position, state.scale_factor)
        };
        let bounds = match given {
            Some(b) => b,
            None => self
                .with_input_handler(|h| {
                    let range = h.selected_text_range(false)?;
                    h.bounds_for_range(range.range)
                })
                .flatten()?,
        };
        Some(POINT {
            x: (bounds.origin.x.0 * scale) as i32,
            y: (bounds.origin.y.0 * scale) as i32 + ((bounds.size.height.0 * scale) as i32 / 2),
        })
    }

    pub(crate) fn handle_ime_position(&self, hwnd: HWND) -> Option<isize> {
        let caret = self.caret_position();
        // SAFETY: the window's own input context, released below.
        unsafe {
            let ctx = ImmGetContext(hwnd);
            if let Some(caret) = caret {
                let composition =
                    COMPOSITIONFORM { dwStyle: CFS_POINT, ptCurrentPos: caret, ..Default::default() };
                let _ = ImmSetCompositionWindow(ctx, &composition);
                let candidate =
                    CANDIDATEFORM { dwStyle: CFS_CANDIDATEPOS, ptCurrentPos: caret, ..Default::default() };
                let _ = ImmSetCandidateWindow(ctx, &candidate);
            }
            let _ = ImmReleaseContext(hwnd, ctx);
        }
        Some(0)
    }

    pub(crate) fn handle_ime_composition(&self, hwnd: HWND, lparam: LPARAM) -> Option<isize> {
        // SAFETY: the window's own input context, released below.
        let ctx = unsafe { ImmGetContext(hwnd) };
        let result = self.ime_composition(ctx, lparam.0 as u32);
        // SAFETY: as above.
        let _ = unsafe { ImmReleaseContext(hwnd, ctx) };
        result
    }

    fn ime_composition(&self, ctx: HIMC, flags: u32) -> Option<isize> {
        if flags == 0 {
            // A Japanese IME sends this with no flags when the composition is gone: drop the composed
            // text. With none under way there is nothing to drop (`replace_text_in_range(None, "")`
            // would delete the selection instead).
            if composition_ended_drops_text(self.with_input_handler(|h| h.marked_text_range()).flatten()) {
                self.with_input_handler(|h| h.replace_text_in_range(None, ""))?;
            }
            return Some(0);
        }
        if flags & GCS_COMPSTR.0 != 0 {
            let text = composition_string(ctx, GCS_COMPSTR)?;
            let caret = (!text.is_empty() && flags & GCS_CURSORPOS.0 != 0)
                .then(|| {
                    // SAFETY: querying the cursor position of a valid context.
                    let pos = unsafe { ImmGetCompositionStringW(ctx, GCS_CURSORPOS, None, 0) };
                    usize::try_from(pos).ok().map(|p| p..p)
                })
                .flatten();
            self.with_input_handler(|h| h.replace_and_mark_text_in_range(None, &text, caret))?;
        }
        if flags & GCS_RESULTSTR.0 != 0 {
            let text = composition_string(ctx, GCS_RESULTSTR)?;
            self.with_input_handler(|h| h.replace_text_in_range(None, &text))?;
            return Some(0);
        }
        None
    }
}

/// Whether a flagless `WM_IME_COMPOSITION` (the composition is gone) removes text: only the marked
/// (composed) text, so only when there is some.
fn composition_ended_drops_text(marked: Option<std::ops::Range<usize>>) -> bool {
    marked.is_some_and(|r| !r.is_empty())
}

/// Moves the IME windows to the app's caret now, if the window has an input context.
pub(crate) fn position_ime(window: &WindowInner) {
    window.handle_ime_position(window.hwnd);
}

fn composition_string(ctx: HIMC, kind: IME_COMPOSITION_STRING) -> Option<String> {
    // SAFETY: the first call sizes the buffer, the second fills it (bytes, UTF-16).
    unsafe {
        let len = ImmGetCompositionStringW(ctx, kind, None, 0);
        if len < 0 {
            return None;
        }
        let mut buffer = vec![0u16; len as usize / 2 + 1];
        ImmGetCompositionStringW(ctx, kind, Some(buffer.as_mut_ptr() as *mut _), len as u32);
        Some(String::from_utf16_lossy(&buffer[..len as usize / 2]))
    }
}

/// Lets Windows turn a key the app did not handle into WM_CHAR / WM_DEADCHAR.
fn translate_message(hwnd: HWND, wparam: WPARAM, lparam: LPARAM) {
    let msg =
        MSG { hwnd, message: WM_KEYDOWN, wParam: wparam, lParam: lparam, time: 0, pt: POINT::default() };
    // SAFETY: a complete MSG for our window.
    let _ = unsafe { TranslateMessage(&msg) };
}

/// One WM_CHAR unit: a character, or half of a surrogate pair waiting for the other half.
fn parse_char(state: &mut WindowState, wparam: WPARAM) -> Option<String> {
    let unit = wparam.loword();
    match unit {
        0xD800..=0xDBFF => {
            state.pending_surrogate = Some(unit);
            None
        }
        0xDC00..=0xDFFF => {
            let high = state.pending_surrogate.take()?;
            String::from_utf16(&[high, unit]).ok()
        }
        _ => {
            state.pending_surrogate = None;
            char::from_u32(unit as u32).filter(|c| !c.is_control()).map(|c| c.to_string())
        }
    }
}

fn key_event(
    hwnd: HWND,
    wparam: WPARAM,
    lparam: LPARAM,
    state: &mut WindowState,
    f: impl FnOnce(Keystroke) -> PlatformInput,
) -> Option<PlatformInput> {
    let vkey = VIRTUAL_KEY(wparam.loword());
    let (modifiers, altgr) = current_modifiers_with_altgr();
    match vkey {
        VK_SHIFT | VK_CONTROL | VK_MENU | VK_LMENU | VK_RMENU | VK_LWIN | VK_RWIN => {
            if state.last_reported_modifiers == Some(modifiers) {
                return None;
            }
            state.last_reported_modifiers = Some(modifiers);
            Some(PlatformInput::ModifiersChanged(ModifiersChangedEvent {
                modifiers,
                capslock: current_capslock(),
            }))
        }
        VK_PACKET => {
            translate_message(hwnd, wparam, lparam);
            None
        }
        VK_CAPITAL => {
            let capslock = current_capslock();
            if state.last_reported_capslock == Some(capslock) {
                return None;
            }
            state.last_reported_capslock = Some(capslock);
            Some(PlatformInput::ModifiersChanged(ModifiersChangedEvent { modifiers, capslock }))
        }
        vkey => {
            let vkey = if vkey == VK_PROCESSKEY {
                // SAFETY: a plain query for our window.
                VIRTUAL_KEY(unsafe { ImmGetVirtualKey(hwnd) } as u16)
            } else {
                vkey
            };
            Some(f(parse_normal_key(vkey, lparam, modifiers, altgr)?))
        }
    }
}

/// GPUI's names for keys whose name does not depend on the layout.
pub(crate) fn immutable_key_name(vkey: VIRTUAL_KEY) -> Option<&'static str> {
    Some(match vkey {
        VK_SPACE => "space",
        VK_BACK => "backspace",
        VK_RETURN => "enter",
        VK_TAB => "tab",
        VK_UP => "up",
        VK_DOWN => "down",
        VK_RIGHT => "right",
        VK_LEFT => "left",
        VK_HOME => "home",
        VK_END => "end",
        VK_PRIOR => "pageup",
        VK_NEXT => "pagedown",
        VK_BROWSER_BACK => "back",
        VK_BROWSER_FORWARD => "forward",
        VK_ESCAPE => "escape",
        VK_INSERT => "insert",
        VK_DELETE => "delete",
        VK_APPS => "menu",
        v if (VK_F1.0..=VK_F24.0).contains(&v.0) => {
            const F: [&str; 24] = [
                "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14",
                "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24",
            ];
            F[(v.0 - VK_F1.0) as usize]
        }
        _ => return None,
    })
}

fn parse_normal_key(
    vkey: VIRTUAL_KEY,
    lparam: LPARAM,
    mut modifiers: Modifiers,
    altgr: bool,
) -> Option<Keystroke> {
    let mut key_char = None;
    let key = match immutable_key_name(vkey) {
        Some(name) => name.to_string(),
        None => {
            let scan_code = (lparam.hiword() & 0xFF) as u32;
            key_char = generate_key_char(
                vkey,
                scan_code,
                modifiers.control || altgr,
                modifiers.shift,
                modifiers.alt || altgr,
            );
            keystroke_key(vkey, scan_code, &mut modifiers)?
        }
    };
    Some(Keystroke { modifiers, key, key_char })
}

/// The key's name: its unshifted character, or for digits and punctuation with shift the shifted
/// character with shift dropped from the modifiers (GPUI's `get_keystroke_key`).
fn keystroke_key(vkey: VIRTUAL_KEY, scan_code: u32, modifiers: &mut Modifiers) -> Option<String> {
    if modifiers.shift && converts_to_shifted_key(vkey) {
        let shifted = generate_key_char(vkey, scan_code, false, true, false)?;
        modifiers.shift = false;
        Some(shifted)
    } else {
        // SAFETY: a plain layout query.
        let data = unsafe { MapVirtualKeyW(vkey.0 as u32, MAPVK_VK_TO_CHAR) };
        if data == 0 {
            return None;
        }
        Some(char::from_u32(data & 0xFFFF)?.to_ascii_lowercase().to_string())
    }
}

fn converts_to_shifted_key(vkey: VIRTUAL_KEY) -> bool {
    matches!(
        vkey,
        VK_OEM_3
            | VK_OEM_MINUS
            | VK_OEM_PLUS
            | VK_OEM_4
            | VK_OEM_5
            | VK_OEM_6
            | VK_OEM_1
            | VK_OEM_7
            | VK_OEM_COMMA
            | VK_OEM_PERIOD
            | VK_OEM_2
            | VK_OEM_102
            | VK_OEM_8
            | VK_ABNT_C1
    ) || (VK_0.0..=VK_9.0).contains(&vkey.0)
}

/// What the key would type with these modifiers (`ToUnicode` without touching the dead-key state).
pub(crate) fn generate_key_char(
    vkey: VIRTUAL_KEY,
    scan_code: u32,
    control: bool,
    shift: bool,
    alt: bool,
) -> Option<String> {
    let mut state = [0u8; 256];
    if control {
        state[VK_CONTROL.0 as usize] = 0x80;
    }
    if shift {
        state[VK_SHIFT.0 as usize] = 0x80;
    }
    if alt {
        state[VK_MENU.0 as usize] = 0x80;
    }
    let mut buffer = [0u16; 8];
    // SAFETY: valid key state and output buffers; flag 4 leaves the keyboard state alone.
    let len = unsafe { ToUnicode(vkey.0 as u32, scan_code, Some(&state), &mut buffer, 1 << 2) };
    match len {
        len if len > 0 => String::from_utf16(&buffer[..len as usize])
            .ok()
            .filter(|s| s.chars().next().is_some_and(|c| !c.is_control())),
        len if len < 0 => String::from_utf16(&buffer[..(-len) as usize]).ok(),
        _ => None,
    }
}

fn key_pressed(vkey: VIRTUAL_KEY) -> bool {
    // SAFETY: a plain query of this thread's key state.
    unsafe { GetKeyState(vkey.0 as i32) < 0 }
}

/// Whether the keyboard layout has AltGr (right Alt + left Ctrl): GPUI's list of layouts.
fn keyboard_uses_altgr() -> bool {
    let mut name = [0u16; KL_NAMELENGTH as usize];
    // SAFETY: a buffer of the documented length.
    if unsafe { GetKeyboardLayoutNameW(&mut name) }.is_err() {
        return false;
    }
    let id = String::from_utf16_lossy(&name[..name.iter().position(|&u| u == 0).unwrap_or(name.len())]);
    let lang = id.get(id.len().saturating_sub(4)..).unwrap_or("").to_ascii_uppercase();
    matches!(
        lang.as_str(),
        "0407"
            | "040C"
            | "040A"
            | "0415"
            | "0413"
            | "0816"
            | "041D"
            | "0414"
            | "040B"
            | "041F"
            | "0419"
            | "0405"
            | "040E"
            | "0424"
            | "041B"
            | "0418"
    )
}

/// The modifiers held now (AltGr is neither Ctrl nor Alt on layouts that have it).
pub(crate) fn current_modifiers() -> Modifiers {
    current_modifiers_with_altgr().0
}

fn current_modifiers_with_altgr() -> (Modifiers, bool) {
    let lmenu = key_pressed(VK_LMENU);
    let rmenu = key_pressed(VK_RMENU);
    let lcontrol = key_pressed(VK_LCONTROL);
    let altgr = rmenu && lcontrol && keyboard_uses_altgr();
    (
        Modifiers {
            control: key_pressed(VK_CONTROL) && !altgr,
            alt: (lmenu || rmenu) && !altgr,
            shift: key_pressed(VK_SHIFT),
            platform: key_pressed(VK_LWIN) || key_pressed(VK_RWIN),
            function: false,
        },
        altgr,
    )
}

/// The caps lock state now.
pub(crate) fn current_capslock() -> Capslock {
    // SAFETY: a plain query of this thread's key state.
    Capslock { on: unsafe { GetKeyState(VK_CAPITAL.0 as i32) } & 1 != 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_names_are_gpuis() {
        assert_eq!(immutable_key_name(VK_RETURN), Some("enter"));
        assert_eq!(immutable_key_name(VK_F11), Some("f11"));
        assert_eq!(immutable_key_name(VK_F24), Some("f24"));
        assert_eq!(immutable_key_name(VK_PRIOR), Some("pageup"));
        assert_eq!(immutable_key_name(VK_A), None);
    }

    /// A bare WM_IME_COMPOSITION with nothing composed left the selection deleted (stress
    /// d5-window-input-1): only composed text goes.
    #[test]
    fn an_ended_composition_drops_only_composed_text() {
        assert!(!composition_ended_drops_text(None));
        assert!(!composition_ended_drops_text(Some(3..3)));
        assert!(composition_ended_drops_text(Some(2..5)));
    }

    #[test]
    fn digits_and_punctuation_convert_with_shift() {
        assert!(converts_to_shifted_key(VK_4));
        assert!(converts_to_shifted_key(VK_OEM_PERIOD));
        assert!(!converts_to_shifted_key(VK_A));
    }
}
