//! Quarter turns and mirrors of a picture, and of everything drawn in its coordinates.

use serde::Deserialize;

use crate::types::Rect;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Turn {
    /// Mirror left and right.
    FlipHorizontal,
    /// Mirror top and bottom.
    FlipVertical,
    /// A quarter turn clockwise.
    RotateRight,
    /// A quarter turn anticlockwise.
    RotateLeft,
}

impl Turn {
    /// The turn that undoes this one.
    pub fn inverse(self) -> Turn {
        match self {
            Turn::RotateRight => Turn::RotateLeft,
            Turn::RotateLeft => Turn::RotateRight,
            mirror => mirror,
        }
    }

    /// Size of a `w` x `h` picture after the turn.
    pub fn size(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            Turn::RotateRight | Turn::RotateLeft => (h, w),
            _ => (w, h),
        }
    }

    /// Where the point `(x, y)` of a `w` x `h` picture goes (continuous coordinates, `0..w` x `0..h`).
    pub fn point(self, x: f64, y: f64, w: f64, h: f64) -> (f64, f64) {
        match self {
            Turn::FlipHorizontal => (w - x, y),
            Turn::FlipVertical => (x, h - y),
            Turn::RotateRight => (h - y, x),
            Turn::RotateLeft => (y, w - x),
        }
    }

    /// The rectangle `r` of a `w` x `h` picture after the turn.
    pub fn rect(self, r: Rect, w: f64, h: f64) -> Rect {
        let (ax, ay) = self.point(r.x0, r.y0, w, h);
        let (bx, by) = self.point(r.x1, r.y1, w, h);
        Rect::new(ax.min(bx), ay.min(by), ax.max(bx), ay.max(by))
    }

    /// Samples of a `w` x `h` picture with `ch` interleaved channels after the turn.
    pub fn pixels<T: Copy>(self, data: &[T], w: usize, h: usize, ch: usize) -> Vec<T> {
        let (nw, nh) = self.size(w, h);
        let mut out = Vec::with_capacity(data.len());
        for ny in 0..nh {
            for nx in 0..nw {
                // The source pixel that lands on (nx, ny): the inverse turn of its centre.
                let (x, y) = match self {
                    Turn::FlipHorizontal => (w - 1 - nx, ny),
                    Turn::FlipVertical => (nx, h - 1 - ny),
                    Turn::RotateRight => (ny, h - 1 - nx),
                    Turn::RotateLeft => (w - 1 - ny, nx),
                };
                out.extend_from_slice(&data[(y * w + x) * ch..(y * w + x + 1) * ch]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_and_its_inverse_give_back_the_picture_and_points_follow_pixels() {
        let (w, h) = (3usize, 2usize);
        let data: Vec<u8> = (0..(w * h) as u8).collect(); // 0 1 2 / 3 4 5
        assert_eq!(Turn::RotateRight.pixels(&data, w, h, 1), vec![3, 0, 4, 1, 5, 2]);
        assert_eq!(Turn::RotateLeft.pixels(&data, w, h, 1), vec![2, 5, 1, 4, 0, 3]);
        assert_eq!(Turn::FlipHorizontal.pixels(&data, w, h, 1), vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(Turn::FlipVertical.pixels(&data, w, h, 1), vec![3, 4, 5, 0, 1, 2]);
        for t in [Turn::RotateRight, Turn::RotateLeft, Turn::FlipHorizontal, Turn::FlipVertical] {
            let (nw, nh) = t.size(w, h);
            let back = t.inverse().pixels(&t.pixels(&data, w, h, 1), nw, nh, 1);
            assert_eq!(back, data, "{t:?}");
            // The centre of each pixel goes where its value goes.
            let turned = t.pixels(&data, w, h, 1);
            for y in 0..h {
                for x in 0..w {
                    let (px, py) = t.point(x as f64 + 0.5, y as f64 + 0.5, w as f64, h as f64);
                    assert_eq!(turned[py as usize * nw + px as usize], data[y * w + x], "{t:?} at {x},{y}");
                }
            }
        }
        let r = Turn::RotateRight.rect(Rect::new(0.0, 0.0, 2.0, 1.0), 3.0, 2.0);
        assert_eq!(r, Rect::new(1.0, 0.0, 2.0, 2.0));
    }
}
