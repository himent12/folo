//! winit keys to GPUI keystrokes: key names and `key_char` as `keystroke.rs` expects them, with the
//! rules of GPUI's Linux backend (`Keystroke::from_xkb`): the key is what the layout types with shift
//! applied, lower-cased; shift is kept only where it changes a letter; control characters are never a
//! `key_char`; a layout without ASCII keys gets the ASCII key at the same position.

use crate::{Capslock, Keystroke, Modifiers};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};

/// GPUI's modifiers from winit's (`platform` is Cmd on macOS, Super elsewhere).
pub(crate) fn modifiers(state: ModifiersState) -> Modifiers {
    Modifiers {
        control: state.control_key(),
        alt: state.alt_key(),
        shift: state.shift_key(),
        platform: state.super_key(),
        function: false,
    }
}

/// One key press or release, as winit reports it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeyInput<'a> {
    /// The key with the layout and shift applied (winit's `logical_key`).
    pub logical: &'a Key,
    /// The key with no modifiers applied (`key_without_modifiers`).
    pub unmodified: &'a Key,
    /// The key's position on the keyboard.
    pub physical: PhysicalKey,
    /// The text the press types (winit's `text`, which ignores Ctrl).
    pub text: Option<&'a str>,
}

/// Which system's rules decide what a key with modifiers types (what winit reports differs per system).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextRules {
    /// AppKit: Option (and Option-Shift) types another character ("ß" for Option-S, which winit reports
    /// only as the key's `text`: `insertText` commits through the IME only while text is marked);
    /// Cmd and Ctrl combinations are commands.
    MacOs,
    /// Win32: AltGr is Ctrl+Alt. winit hides it on a layout with AltGr when right Alt is held, but left
    /// Ctrl + left Alt still arrive as Ctrl+Alt, and winit gives them text only where the layout has a
    /// character for them ("€" for Ctrl+Alt+E on a German layout; none on a US one), as Windows sends
    /// `WM_CHAR`, which GPUI types.
    Windows,
    /// xkb (Linux, BSD): AltGr is Level3, which winit does not report as a modifier, so its characters
    /// arrive with at most Shift; Ctrl+Alt is a command whose text is the plain character.
    Xkb,
}

impl TextRules {
    /// The rules of the system this is built for.
    pub(crate) const CURRENT: Self = if cfg!(target_os = "macos") {
        Self::MacOs
    } else if cfg!(windows) {
        Self::Windows
    } else {
        Self::Xkb
    };

    /// Whether a key held with `modifiers` keeps its text as the keystroke's `key_char`.
    fn has_key_char(self, modifiers: &Modifiers) -> bool {
        match self {
            Self::MacOs => !modifiers.control && !modifiers.platform,
            Self::Windows => !modifiers.control || modifiers.alt,
            Self::Xkb => !modifiers.control,
        }
    }

    /// Whether a key down the app did not handle types its `key_char` with `modifiers` held (the
    /// keystroke's, so Shift only where it changed a letter).
    pub(crate) fn types_text(self, modifiers: &Modifiers) -> bool {
        if modifiers.platform || modifiers.function {
            return false;
        }
        match self {
            Self::MacOs => !modifiers.control,
            Self::Windows => modifiers.control == modifiers.alt,
            Self::Xkb => !modifiers.control && !modifiers.alt,
        }
    }
}

/// The GPUI keystroke for a key press with `modifiers` held.
pub(crate) fn keystroke(input: KeyInput<'_>, modifiers: Modifiers) -> Keystroke {
    keystroke_for(TextRules::CURRENT, input, modifiers)
}

/// [`keystroke`] under `rules`.
fn keystroke_for(rules: TextRules, input: KeyInput<'_>, mut modifiers: Modifiers) -> Keystroke {
    // On macOS Option changes the character (option-s types "ß"); the key is still "s".
    let key = if rules == TextRules::MacOs && modifiers.alt { input.unmodified } else { input.logical };
    let key = key_name(key, input.physical, modifiers.shift);

    // Shift is part of a keystroke only where it makes a letter upper case ("shift-a", "shift-tab"), not
    // where it picked a symbol ("!" rather than "shift-1").
    if modifiers.shift && key.chars().count() == 1 && key.to_lowercase() == key.to_uppercase() {
        modifiers.shift = false;
    }

    let has_key_char = rules.has_key_char(&modifiers);
    let key_char = input
        .text
        .filter(|text| has_key_char && text.chars().next().is_some_and(|c| c as u32 >= 32 && c as u32 != 127))
        .map(str::to_owned);

    Keystroke { modifiers, key, key_char }
}

/// Whether this key toggles caps lock.
pub(crate) fn is_capslock(key: &Key) -> bool {
    matches!(key, Key::Named(NamedKey::CapsLock))
}

/// Flips the caps lock state (winit does not report it; it is tracked from the key).
pub(crate) fn toggle(capslock: Capslock) -> Capslock {
    Capslock { on: !capslock.on }
}

fn key_name(key: &Key, physical: PhysicalKey, shift: bool) -> String {
    match key {
        Key::Named(named) => named_key(*named),
        Key::Character(text) => {
            let mut chars = text.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_graphic() => c.to_ascii_lowercase().to_string(),
                _ => ascii_at(physical, shift).map(str::to_owned).unwrap_or_else(|| text.to_lowercase()),
            }
        }
        Key::Dead(c) => ascii_at(physical, shift)
            .map(str::to_owned)
            .or_else(|| c.map(|c| c.to_string()))
            .unwrap_or_else(|| "dead".into()),
        Key::Unidentified(_) => ascii_at(physical, shift).unwrap_or("unidentified").to_owned(),
    }
}

/// GPUI's name of a named key.
fn named_key(named: NamedKey) -> String {
    let name = match named {
        NamedKey::ArrowLeft => "left",
        NamedKey::ArrowRight => "right",
        NamedKey::ArrowUp => "up",
        NamedKey::ArrowDown => "down",
        NamedKey::PageUp => "pageup",
        NamedKey::PageDown => "pagedown",
        NamedKey::BrowserBack => "back",
        NamedKey::BrowserForward => "forward",
        NamedKey::Super | NamedKey::Meta => "platform",
        NamedKey::Fn => "function",
        NamedKey::Control => "control",
        NamedKey::ContextMenu => "menu",
        // Enter, Tab, Space, Backspace, Delete, Escape, Home, End, Insert, F1..F35, Shift, Alt, CapsLock,
        // Copy, Cut, Paste, …: their own names in lower case.
        other => return format!("{other:?}").to_lowercase(),
    };
    name.to_owned()
}

/// The ASCII character a US layout has at `physical` (GPUI's `guess_ascii`), for layouts whose keys
/// are not ASCII.
fn ascii_at(physical: PhysicalKey, shift: bool) -> Option<&'static str> {
    let PhysicalKey::Code(code) = physical else { return None };
    const LETTERS: [&str; 26] = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
    ];
    const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
    const SHIFTED_DIGITS: [&str; 10] = [")", "!", "@", "#", "$", "%", "^", "&", "*", "("];
    let letter = |i: usize| Some(LETTERS[i]);
    let digit = |i: usize| Some(if shift { SHIFTED_DIGITS[i] } else { DIGITS[i] });
    let symbol = |plain: &'static str, shifted: &'static str| Some(if shift { shifted } else { plain });
    match code {
        KeyCode::KeyA => letter(0),
        KeyCode::KeyB => letter(1),
        KeyCode::KeyC => letter(2),
        KeyCode::KeyD => letter(3),
        KeyCode::KeyE => letter(4),
        KeyCode::KeyF => letter(5),
        KeyCode::KeyG => letter(6),
        KeyCode::KeyH => letter(7),
        KeyCode::KeyI => letter(8),
        KeyCode::KeyJ => letter(9),
        KeyCode::KeyK => letter(10),
        KeyCode::KeyL => letter(11),
        KeyCode::KeyM => letter(12),
        KeyCode::KeyN => letter(13),
        KeyCode::KeyO => letter(14),
        KeyCode::KeyP => letter(15),
        KeyCode::KeyQ => letter(16),
        KeyCode::KeyR => letter(17),
        KeyCode::KeyS => letter(18),
        KeyCode::KeyT => letter(19),
        KeyCode::KeyU => letter(20),
        KeyCode::KeyV => letter(21),
        KeyCode::KeyW => letter(22),
        KeyCode::KeyX => letter(23),
        KeyCode::KeyY => letter(24),
        KeyCode::KeyZ => letter(25),
        KeyCode::Digit0 => digit(0),
        KeyCode::Digit1 => digit(1),
        KeyCode::Digit2 => digit(2),
        KeyCode::Digit3 => digit(3),
        KeyCode::Digit4 => digit(4),
        KeyCode::Digit5 => digit(5),
        KeyCode::Digit6 => digit(6),
        KeyCode::Digit7 => digit(7),
        KeyCode::Digit8 => digit(8),
        KeyCode::Digit9 => digit(9),
        KeyCode::Minus => symbol("-", "_"),
        KeyCode::Equal => symbol("=", "+"),
        KeyCode::BracketLeft => symbol("[", "{"),
        KeyCode::BracketRight => symbol("]", "}"),
        KeyCode::Backslash => symbol("\\", "|"),
        KeyCode::Semicolon => symbol(";", ":"),
        KeyCode::Quote => symbol("'", "\""),
        KeyCode::Backquote => symbol("`", "~"),
        KeyCode::Comma => symbol(",", "<"),
        KeyCode::Period => symbol(".", ">"),
        KeyCode::Slash => symbol("/", "?"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::SmolStr;

    fn ch(s: &str) -> Key {
        Key::Character(SmolStr::new(s))
    }

    fn press(logical: Key, unmodified: Key, code: KeyCode, text: Option<&str>, mods: &str) -> Keystroke {
        let modifiers = Modifiers {
            control: mods.contains("ctrl"),
            alt: mods.contains("alt"),
            shift: mods.contains("shift"),
            platform: mods.contains("cmd"),
            function: false,
        };
        let input =
            KeyInput { logical: &logical, unmodified: &unmodified, physical: PhysicalKey::Code(code), text };
        keystroke(input, modifiers)
    }

    fn named(n: NamedKey) -> Key {
        Key::Named(n)
    }

    #[test]
    fn letters_and_shift() {
        let a = press(ch("a"), ch("a"), KeyCode::KeyA, Some("a"), "");
        assert_eq!((a.unparse(), a.key_char.as_deref()), ("a".into(), Some("a")));
        let shift_a = press(ch("A"), ch("a"), KeyCode::KeyA, Some("A"), "shift");
        assert_eq!((shift_a.unparse(), shift_a.key_char.as_deref()), ("shift-a".into(), Some("A")));
        // Shift that picks a symbol is not part of the keystroke.
        let bang = press(ch("!"), ch("1"), KeyCode::Digit1, Some("!"), "shift");
        assert_eq!((bang.unparse(), bang.key_char.as_deref()), ("!".into(), Some("!")));
    }

    #[test]
    fn control_combinations_have_no_key_char() {
        let ctrl_k = press(ch("k"), ch("k"), KeyCode::KeyK, Some("k"), "ctrl");
        assert_eq!((ctrl_k.unparse(), ctrl_k.key_char), ("ctrl-k".into(), None));
        let ctrl_comma = press(ch(","), ch(","), KeyCode::Comma, Some(","), "ctrl");
        assert_eq!(ctrl_comma.unparse(), "ctrl-,");
        assert!(
            ctrl_comma.should_match(&crate::KeybindingKeystroke::from_keystroke(
                Keystroke::parse("ctrl-,").unwrap()
            ))
        );
    }

    #[test]
    fn named_keys_use_gpui_names() {
        let cases = [
            (NamedKey::Escape, Some("\u{1b}"), "", "escape", None),
            (NamedKey::Enter, Some("\r"), "", "enter", None),
            (NamedKey::Tab, Some("\t"), "shift", "shift-tab", None),
            (NamedKey::F11, None, "", "f11", None),
            (NamedKey::ArrowLeft, None, "alt", "alt-left", None),
            (NamedKey::ArrowUp, None, "", "up", None),
            (NamedKey::PageUp, None, "", "pageup", None),
            (NamedKey::PageDown, None, "", "pagedown", None),
            (NamedKey::Backspace, Some("\u{8}"), "", "backspace", None),
            (NamedKey::Delete, Some("\u{7f}"), "", "delete", None),
            (NamedKey::Home, None, "", "home", None),
            (NamedKey::Space, Some(" "), "", "space", Some(" ")),
        ];
        for (key, text, mods, want, key_char) in cases {
            let k = press(named(key), named(key), KeyCode::Escape, text, mods);
            assert_eq!(k.unparse(), want, "{key:?}");
            assert_eq!(k.key_char.as_deref(), key_char, "{key:?}");
            assert_eq!(Keystroke::parse(want).unwrap().key, k.key, "{key:?} parses back");
        }
        assert_eq!(
            press(named(NamedKey::Super), named(NamedKey::Super), KeyCode::SuperLeft, None, "cmd").key,
            "platform"
        );
    }

    #[test]
    fn non_ascii_layouts_map_to_the_ascii_key() {
        // Cyrillic "ф" sits on the A key: the key is "a", the typed char stays.
        let ef = press(ch("ф"), ch("ф"), KeyCode::KeyA, Some("ф"), "ctrl");
        assert_eq!(ef.unparse(), "ctrl-a");
        let ef = press(ch("ф"), ch("ф"), KeyCode::KeyA, Some("ф"), "");
        assert_eq!((ef.key.as_str(), ef.key_char.as_deref()), ("a", Some("ф")));
        // A key with no ASCII position keeps its character, lower-cased.
        let x = press(ch("É"), ch("é"), KeyCode::IntlBackslash, Some("É"), "shift");
        assert_eq!((x.key.as_str(), x.modifiers.shift), ("é", true));
    }

    /// What a key the app did not handle types under `rules`: its `key_char`, if the modifiers type.
    fn typed(
        rules: TextRules,
        logical: Key,
        unmodified: Key,
        code: KeyCode,
        text: &str,
        mods: &str,
    ) -> Option<String> {
        let modifiers = Modifiers {
            control: mods.contains("ctrl"),
            alt: mods.contains("alt"),
            shift: mods.contains("shift"),
            platform: mods.contains("cmd"),
            function: false,
        };
        let physical = PhysicalKey::Code(code);
        let input = KeyInput { logical: &logical, unmodified: &unmodified, physical, text: Some(text) };
        let keystroke = keystroke_for(rules, input, modifiers);
        keystroke.key_char.filter(|_| rules.types_text(&keystroke.modifiers))
    }

    #[test]
    fn option_types_characters_on_macos() {
        use TextRules::MacOs;
        // Option-S on a US layout: winit's `text` is "ß" (the event's characters); the key stays "s".
        let (logical, unmodified) = (ch("ß"), ch("s"));
        let physical = PhysicalKey::Code(KeyCode::KeyS);
        let input = KeyInput { logical: &logical, unmodified: &unmodified, physical, text: Some("ß") };
        let option_s = keystroke_for(MacOs, input, Modifiers { alt: true, ..Modifiers::default() });
        assert_eq!((option_s.unparse(), option_s.key_char.as_deref()), ("alt-s".into(), Some("ß")));
        assert_eq!(typed(MacOs, ch("ß"), ch("s"), KeyCode::KeyS, "ß", "alt").as_deref(), Some("ß"));
        // Option-Shift-2 types "€".
        assert_eq!(typed(MacOs, ch("€"), ch("2"), KeyCode::Digit2, "€", "alt shift").as_deref(), Some("€"));
        assert_eq!(typed(MacOs, ch("a"), ch("a"), KeyCode::KeyA, "a", "").as_deref(), Some("a"));
        // Cmd and Ctrl combinations are commands, with or without Option.
        assert_eq!(typed(MacOs, ch("s"), ch("s"), KeyCode::KeyS, "s", "cmd"), None);
        assert_eq!(typed(MacOs, ch("ß"), ch("s"), KeyCode::KeyS, "ß", "cmd alt"), None);
        assert_eq!(typed(MacOs, ch("s"), ch("s"), KeyCode::KeyS, "s", "ctrl"), None);
        assert_eq!(typed(MacOs, ch("ß"), ch("s"), KeyCode::KeyS, "ß", "ctrl alt"), None);
    }

    #[test]
    fn altgr_types_characters_on_windows_and_xkb() {
        use TextRules::{Windows, Xkb};
        // Windows: left Ctrl + left Alt + E on a German layout is AltGr+E, "€" (the key is "e").
        assert_eq!(typed(Windows, ch("€"), ch("e"), KeyCode::KeyE, "€", "ctrl alt").as_deref(), Some("€"));
        // AltGr+Q is "@", an ASCII character.
        assert_eq!(typed(Windows, ch("@"), ch("q"), KeyCode::KeyQ, "@", "ctrl alt").as_deref(), Some("@"));
        // xkb: AltGr is Level3, not a modifier winit reports; its characters arrive with no modifiers.
        assert_eq!(typed(Xkb, ch("€"), ch("e"), KeyCode::KeyE, "€", "").as_deref(), Some("€"));
        assert_eq!(typed(Xkb, ch("{"), ch("7"), KeyCode::Digit7, "{", "").as_deref(), Some("{"));
        for rules in [Windows, Xkb] {
            // Plain Alt and Ctrl combinations are commands (winit's text ignores Ctrl: Ctrl+K has "k").
            assert_eq!(typed(rules, ch("k"), ch("k"), KeyCode::KeyK, "k", "alt"), None, "{rules:?}");
            assert_eq!(typed(rules, ch("k"), ch("k"), KeyCode::KeyK, "k", "ctrl"), None, "{rules:?}");
            assert_eq!(typed(rules, ch("K"), ch("k"), KeyCode::KeyK, "K", "ctrl shift"), None, "{rules:?}");
            assert_eq!(typed(rules, ch("k"), ch("k"), KeyCode::KeyK, "k", "cmd"), None, "{rules:?}");
            let shift_a = typed(rules, ch("A"), ch("a"), KeyCode::KeyA, "A", "shift");
            assert_eq!(shift_a.as_deref(), Some("A"), "{rules:?}");
        }
        // xkb's Ctrl+Alt is a command chord: its text is the plain character.
        assert_eq!(typed(Xkb, ch("k"), ch("k"), KeyCode::KeyK, "k", "ctrl alt"), None);
    }

    #[test]
    fn modifier_state_and_capslock() {
        let m = modifiers(ModifiersState::SHIFT | ModifiersState::SUPER);
        assert!(m.shift && m.platform && !m.control && !m.alt);
        assert!(is_capslock(&named(NamedKey::CapsLock)) && !is_capslock(&ch("a")));
        assert!(toggle(Capslock { on: false }).on);
    }
}
