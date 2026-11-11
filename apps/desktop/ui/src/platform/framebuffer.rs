//! `Framebuffer`: a window's pixels on the CPU (`ui/DESIGN.md`, Units and colour).

/// A window's pixels: `width` x `height` device pixels, rows top-down, each pixel `0x00RRGGBB`
/// (8-bit gamma-space channels, opaque).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Framebuffer {
    /// Width in device pixels.
    pub width: u32,
    /// Height in device pixels.
    pub height: u32,
    /// `width * height` pixels, row-major, top row first.
    pub pixels: Vec<u32>,
}

impl Framebuffer {
    /// A black buffer of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, pixels: vec![0; width as usize * height as usize] }
    }

    /// Changes the size, keeping the allocation where it can. The content is undefined afterwards
    /// (black where the buffer grew); the renderer repaints after `invalidate`.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pixels.resize(width as usize * height as usize, 0);
    }

    /// Fills every pixel with `color` (`0x00RRGGBB`).
    pub fn fill(&mut self, color: u32) {
        self.pixels.fill(color);
    }

    /// Row `y`. Panics if `y >= height`.
    pub fn row(&self, y: u32) -> &[u32] {
        assert!(y < self.height, "row {y} out of {}", self.height);
        let w = self.width as usize;
        &self.pixels[y as usize * w..(y as usize + 1) * w]
    }

    /// Row `y`, mutable. Panics if `y >= height`.
    pub fn row_mut(&mut self, y: u32) -> &mut [u32] {
        assert!(y < self.height, "row {y} out of {}", self.height);
        let w = self.width as usize;
        &mut self.pixels[y as usize * w..(y as usize + 1) * w]
    }

    /// The pixel at (`x`, `y`). Panics outside the buffer.
    pub fn pixel(&self, x: u32, y: u32) -> u32 {
        assert!(x < self.width, "column {x} out of {}", self.width);
        self.row(y)[x as usize]
    }

    /// Sets the pixel at (`x`, `y`). Panics outside the buffer.
    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32) {
        assert!(x < self.width, "column {x} out of {}", self.width);
        self.row_mut(y)[x as usize] = color;
    }

    /// Whether the buffer has no pixels.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }
}

/// Packs 8-bit channels into a framebuffer pixel (`0x00RRGGBB`).
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

/// Unpacks a framebuffer pixel into `[r, g, b]`.
pub fn unpack_rgb(pixel: u32) -> [u8; 3] {
    [(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_top_down_and_row_major() {
        let mut fb = Framebuffer::new(3, 2);
        assert_eq!(fb.pixels.len(), 6);
        fb.set_pixel(2, 1, pack_rgb(1, 2, 3));
        fb.row_mut(0)[1] = 0x00ff_0000;
        assert_eq!(fb.pixels[5], 0x0001_0203);
        assert_eq!(fb.pixels[1], 0x00ff_0000);
        assert_eq!(fb.row(1), &[0, 0, 0x0001_0203]);
        assert_eq!(fb.pixel(1, 0), 0x00ff_0000);
        assert_eq!(unpack_rgb(fb.pixel(2, 1)), [1, 2, 3]);
    }

    #[test]
    fn resize_changes_length() {
        let mut fb = Framebuffer::new(4, 4);
        fb.fill(7);
        fb.resize(2, 3);
        assert_eq!((fb.width, fb.height, fb.pixels.len()), (2, 3, 6));
        assert_eq!(fb.row(2).len(), 2);
    }

    #[test]
    #[should_panic]
    fn row_out_of_range_panics() {
        Framebuffer::new(2, 2).row(2);
    }
}
