//! Minimal, strict, zero-dependency XML parser for Pascal VOC annotation files.
//!
//! Supports UTF-8 BOM, XML declarations (`<?...?>`), comments (`<!--...-->`), `<!DOCTYPE ...>`,
//! `<![CDATA[...]]>`, attributes, self-closing tags, and standard + numeric XML character entities
//! (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, `&#NNN;`, `&#xHHH;`). Reports line-numbered errors
//! on malformed, truncated, or mismatched markup.

/// Maximum element nesting depth allowed while parsing (guards against pathological inputs).
const MAX_DEPTH: usize = 64;

/// A parsed XML element node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Element {
    /// Tag name (e.g. `annotation`, `object`, `bndbox`, `xmin`).
    pub name: String,
    /// Concatenated decoded character data directly inside this element.
    pub text: String,
    /// Direct child elements, in document order.
    pub children: Vec<Element>,
}

impl Element {
    /// First direct child with tag `name`.
    pub fn child(&self, name: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Trimmed text of the first direct child with tag `name`.
    pub fn child_text(&self, name: &str) -> Option<&str> {
        self.child(name).map(|c| c.text.trim())
    }

    /// All direct children with tag `name`, in document order.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Element> {
        self.children.iter().filter(move |c| c.name == name)
    }
}

/// Parses one XML document into its root [`Element`].
pub(crate) fn parse_document(input: &str) -> Result<Element, String> {
    let text = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut p = Parser::new(text);
    p.skip_prolog()?;
    if p.is_eof() {
        return Err(p.err("empty XML document"));
    }
    if p.peek() != Some(b'<') {
        return Err(p.err("expected root element '<'"));
    }
    let root = p.parse_element(0)?;
    p.skip_epilog()?;
    if !p.is_eof() {
        return Err(p.err("unexpected trailing content after root element"));
    }
    Ok(root)
}

/// Escapes `text` for safe inclusion as XML element text content.
pub(crate) fn escape_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Decodes XML character entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, `&#...;`, `&#x...;`).
pub(crate) fn decode_entities(raw: &str, line: usize) -> Result<String, String> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    let mut cur_line = line;
    while let Some(amp) = rest.find('&') {
        let prefix = &rest[..amp];
        cur_line += prefix.bytes().filter(|&b| b == b'\n').count();
        out.push_str(prefix);
        let after = &rest[amp + 1..];
        let Some(semi) = after.find(';') else {
            return Err(format!("line {cur_line}: unterminated XML entity starting with '&'"));
        };
        let entity = &after[..semi];
        match entity {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            num if let Some(hex) = num.strip_prefix("#x").or_else(|| num.strip_prefix("#X")) => {
                if hex.is_empty() {
                    return Err(format!("line {cur_line}: empty hex character reference '&#x;'"));
                }
                let code = u32::from_str_radix(hex, 16)
                    .map_err(|_| format!("line {cur_line}: invalid hex character reference '&{num};'"))?;
                let ch = valid_xml_char(code)
                    .ok_or_else(|| format!("line {cur_line}: invalid XML code point '&{num};'"))?;
                out.push(ch);
            }
            num if let Some(dec) = num.strip_prefix('#') => {
                if dec.is_empty() {
                    return Err(format!("line {cur_line}: empty numeric character reference '&#;'"));
                }
                let code = dec
                    .parse::<u32>()
                    .map_err(|_| format!("line {cur_line}: invalid numeric character reference '&{num};'"))?;
                let ch = valid_xml_char(code)
                    .ok_or_else(|| format!("line {cur_line}: invalid XML code point '&{num};'"))?;
                out.push(ch);
            }
            other => {
                return Err(format!("line {cur_line}: unknown XML entity '&{other};'"));
            }
        }
        rest = &after[semi + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Validates a Unicode scalar value against XML 1.0 `Char` (forbids NUL and invalid control chars).
fn valid_xml_char(code: u32) -> Option<char> {
    let ok = matches!(code, 0x9 | 0xA | 0xD | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF);
    if ok { char::from_u32(code) } else { None }
}

struct Parser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, bytes: src.as_bytes(), pos: 0, line: 1 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn starts_with(&self, s: &str) -> bool {
        self.rest().starts_with(s)
    }

    fn advance(&mut self, n: usize) {
        for &b in &self.bytes[self.pos..self.pos + n] {
            if b == b'\n' {
                self.line += 1;
            }
        }
        self.pos += n;
    }

    fn err(&self, msg: impl std::fmt::Display) -> String {
        format!("line {}: {msg}", self.line)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b.is_ascii_whitespace() {
                self.advance(1);
            } else {
                break;
            }
        }
    }

    fn skip_prolog(&mut self) -> Result<(), String> {
        loop {
            self.skip_ws();
            if self.starts_with("<?") {
                self.skip_pi()?;
            } else if self.starts_with("<!--") {
                self.skip_comment()?;
            } else if self.starts_with("<!DOCTYPE") {
                self.skip_doctype()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn skip_epilog(&mut self) -> Result<(), String> {
        loop {
            self.skip_ws();
            if self.starts_with("<?") {
                self.skip_pi()?;
            } else if self.starts_with("<!--") {
                self.skip_comment()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn skip_pi(&mut self) -> Result<(), String> {
        let start_line = self.line;
        self.advance(2);
        let Some(end) = self.rest().find("?>") else {
            return Err(format!("line {start_line}: unclosed processing instruction '<?'"));
        };
        self.advance(end + 2);
        Ok(())
    }

    fn skip_comment(&mut self) -> Result<(), String> {
        let start_line = self.line;
        self.advance(4);
        let Some(end) = self.rest().find("-->") else {
            return Err(format!("line {start_line}: unclosed XML comment '<!--'"));
        };
        if self.rest()[..end].contains("--") {
            return Err(format!("line {start_line}: '--' is not permitted inside an XML comment"));
        }
        self.advance(end + 3);
        Ok(())
    }

    fn skip_doctype(&mut self) -> Result<(), String> {
        let start_line = self.line;
        self.advance("<!DOCTYPE".len());
        let mut bracket_depth = 0usize;
        let mut quote: Option<u8> = None;
        while let Some(b) = self.peek() {
            self.advance(1);
            if let Some(q) = quote {
                if b == q {
                    quote = None;
                }
                continue;
            }
            match b {
                b'"' | b'\'' => quote = Some(b),
                b'[' => bracket_depth += 1,
                b']' => bracket_depth = bracket_depth.saturating_sub(1),
                b'>' if bracket_depth == 0 => return Ok(()),
                _ => {}
            }
        }
        Err(format!("line {start_line}: unclosed <!DOCTYPE declaration"))
    }

    fn parse_name(&mut self) -> Result<&'a str, String> {
        let start = self.pos;
        let Some(first) = self.rest().chars().next() else {
            return Err(self.err("unexpected end of XML while reading tag name"));
        };
        if !is_name_start(first) {
            return Err(self.err(format!("invalid XML name starting with {first:?}")));
        }
        self.advance(first.len_utf8());
        while let Some(ch) = self.rest().chars().next() {
            if is_name_char(ch) {
                self.advance(ch.len_utf8());
            } else {
                break;
            }
        }
        Ok(&self.src[start..self.pos])
    }

    fn skip_attributes(&mut self) -> Result<(), String> {
        loop {
            let had_ws = self.peek().is_some_and(|b| b.is_ascii_whitespace());
            self.skip_ws();
            match self.peek() {
                Some(b'>') | Some(b'/') => return Ok(()),
                None => return Err(self.err("unclosed start tag")),
                Some(_) if !had_ws => {
                    return Err(self.err("expected whitespace before attribute in start tag"));
                }
                Some(_) => {
                    let _attr_name = self.parse_name()?;
                    self.skip_ws();
                    if self.peek() != Some(b'=') {
                        return Err(self.err("expected '=' after attribute name"));
                    }
                    self.advance(1);
                    self.skip_ws();
                    let Some(q @ (b'"' | b'\'')) = self.peek() else {
                        return Err(self.err("expected quoted attribute value"));
                    };
                    let val_line = self.line;
                    self.advance(1);
                    let start = self.pos;
                    let mut closed = false;
                    while let Some(b) = self.peek() {
                        if b == q {
                            let raw_val = &self.src[start..self.pos];
                            if raw_val.contains('<') {
                                return Err(format!("line {val_line}: unescaped '<' inside attribute value"));
                            }
                            let _ = decode_entities(raw_val, val_line)?;
                            self.advance(1);
                            closed = true;
                            break;
                        }
                        self.advance(1);
                    }
                    if !closed {
                        return Err(format!("line {val_line}: unclosed attribute value"));
                    }
                }
            }
        }
    }

    fn parse_element(&mut self, depth: usize) -> Result<Element, String> {
        if depth >= MAX_DEPTH {
            return Err(self.err(format!("XML nesting exceeds maximum depth of {MAX_DEPTH}")));
        }
        let open_line = self.line;
        debug_assert_eq!(self.peek(), Some(b'<'));
        self.advance(1);
        let name = self.parse_name()?.to_string();
        self.skip_attributes()?;
        if self.starts_with("/>") {
            self.advance(2);
            return Ok(Element { name, text: String::new(), children: Vec::new() });
        }
        if self.peek() != Some(b'>') {
            return Err(self.err(format!("expected '>' to close start tag <{name}>")));
        }
        self.advance(1);

        let mut text = String::new();
        let mut children = Vec::new();
        loop {
            if self.is_eof() {
                return Err(format!("line {open_line}: unclosed tag <{name}>"));
            }
            if self.starts_with("</") {
                self.advance(2);
                let close_name = self.parse_name()?;
                self.skip_ws();
                if self.peek() != Some(b'>') {
                    return Err(self.err(format!("expected '>' in closing tag </{close_name}>")));
                }
                self.advance(1);
                if close_name != name {
                    return Err(self.err(format!(
                        "mismatched closing tag </{close_name}>, expected </{name}> (opened on line {open_line})"
                    )));
                }
                return Ok(Element { name, text, children });
            } else if self.starts_with("<!--") {
                self.skip_comment()?;
            } else if self.starts_with("<![CDATA[") {
                let cdata_line = self.line;
                self.advance("<![CDATA[".len());
                let Some(end) = self.rest().find("]]>") else {
                    return Err(format!("line {cdata_line}: unclosed <![CDATA[ section"));
                };
                text.push_str(&self.rest()[..end]);
                self.advance(end + 3);
            } else if self.starts_with("<?") {
                self.skip_pi()?;
            } else if self.peek() == Some(b'<') {
                children.push(self.parse_element(depth + 1)?);
            } else {
                let chunk_line = self.line;
                let start = self.pos;
                let end = self.rest().find('<').unwrap_or(self.rest().len());
                self.advance(end);
                let raw = &self.src[start..self.pos];
                if raw.contains("]]>") {
                    return Err(format!("line {chunk_line}: ']]>' is not allowed in element text"));
                }
                text.push_str(&decode_entities(raw, chunk_line)?);
            }
        }
    }
}

fn is_name_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == ':'
}

fn is_name_char(c: char) -> bool {
    is_name_start(c) || c.is_ascii_digit() || matches!(c, '-' | '.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_voc_xml_with_entities_cdata_comments_and_bom() {
        let xml = "\u{feff}<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
            <!-- sample comment -->\n\
            <annotation verified=\"yes\">\n\
                <filename>cat &amp; dog &#x23;1.jpg</filename>\n\
                <size><width>640</width><height>480</height><depth>3</depth></size>\n\
                <segmented/>\n\
                <object>\n\
                    <name><![CDATA[a < b & c]]></name>\n\
                    <difficult>0</difficult>\n\
                    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>220</ymax></bndbox>\n\
                </object>\n\
            </annotation>\n";
        let root = parse_document(xml).unwrap();
        assert_eq!(root.name, "annotation");
        assert_eq!(root.child_text("filename"), Some("cat & dog #1.jpg"));
        let obj = root.child("object").unwrap();
        assert_eq!(obj.child_text("name"), Some("a < b & c"));
        assert_eq!(obj.child("bndbox").unwrap().child_text("xmax"), Some("110"));
    }

    #[test]
    fn entity_decoding_and_escaping() {
        assert_eq!(decode_entities("&lt;&gt;&amp;&quot;&apos;&#65;&#x42;", 1).unwrap(), "<>&\"'AB");
        assert_eq!(escape_text("a < b & \"c\" 'd' > e"), "a &lt; b &amp; &quot;c&quot; &apos;d&apos; &gt; e");
        assert!(decode_entities("&unknown;", 3).unwrap_err().contains("line 3"));
        assert!(decode_entities("&#0;", 1).is_err());
        assert!(decode_entities("&amp", 2).unwrap_err().contains("line 2"));
    }

    #[test]
    fn rejects_malformed_xml_with_clear_line_numbers() {
        assert!(parse_document("").unwrap_err().contains("empty"));
        assert!(
            parse_document("<annotation>\n<xmin>1</xmax>\n</annotation>").unwrap_err().contains("mismatched")
        );
        assert!(parse_document("<annotation><object></annotation>").unwrap_err().contains("mismatched"));
        assert!(parse_document("<annotation><size>").unwrap_err().contains("unclosed"));
        assert!(parse_document("<annotation><!-- unclosed").unwrap_err().contains("unclosed XML comment"));
        assert!(parse_document("<annotation/> trailing").unwrap_err().contains("trailing"));
    }
}
