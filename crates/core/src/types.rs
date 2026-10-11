//! Plain pixel buffers shared by the whole core. No external image types leak out.

use serde::{Deserialize, Serialize};

/// 8-bit RGB image, row-major, 3 bytes per pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// 16-bit RGB image (a 16-bit source kept at full precision), 3 samples per pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Image16 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u16>,
}

/// 16-bit RGBA image with straight alpha, 4 samples per pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgba16 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u16>,
}

/// 8-bit single-channel mask (255 = foreground).
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// 8-bit RGBA image with straight (non-premultiplied) alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgba {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0; w * h * 3] }
    }
    pub fn filled(w: usize, h: usize, rgb: [u8; 3]) -> Self {
        let mut data = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            data.extend_from_slice(&rgb);
        }
        Self { w, h, data }
    }
    pub fn from_raw(w: usize, h: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), w * h * 3, "RGB buffer size mismatch");
        Self { w, h, data }
    }
    pub fn gray(&self) -> Mask {
        let data =
            self.data.chunks_exact(3).map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000) as u8).collect();
        Mask { w: self.w, h: self.h, data }
    }
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0; w * h] }
    }
    pub fn filled(w: usize, h: usize, v: u8) -> Self {
        Self { w, h, data: vec![v; w * h] }
    }
    pub fn from_raw(w: usize, h: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), w * h, "mask buffer size mismatch");
        Self { w, h, data }
    }
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }
    pub fn is_empty_mask(&self) -> bool {
        self.data.iter().all(|&v| v == 0)
    }
}

impl Rgba {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0; w * h * 4] }
    }
}

/// Axis-aligned rectangle in image coordinates, `x1`/`y1` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self { x0, y0, x1, y1 }
    }
    pub fn width(&self) -> f64 {
        self.x1 - self.x0
    }
    pub fn height(&self) -> f64 {
        self.y1 - self.y0
    }
}
