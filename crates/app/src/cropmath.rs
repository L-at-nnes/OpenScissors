//! Crop rectangle editing with an optional aspect ratio, in image coordinates.

use openscissors_core::types::Rect;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    N,
    S,
    E,
    W,
    NE,
    NW,
    SE,
    SW,
    Move,
}

impl Handle {
    fn has(self, side: char) -> bool {
        let s = match self {
            Handle::N => "n",
            Handle::S => "s",
            Handle::E => "e",
            Handle::W => "w",
            Handle::NE => "ne",
            Handle::NW => "nw",
            Handle::SE => "se",
            Handle::SW => "sw",
            Handle::Move => "",
        };
        s.contains(side)
    }

    fn is_side(self) -> bool {
        matches!(self, Handle::N | Handle::S | Handle::E | Handle::W)
    }

    /// Cursor code of the interface for this handle.
    pub fn cursor(self) -> i32 {
        match self {
            Handle::Move => 5,
            Handle::N | Handle::S => 6,
            Handle::E | Handle::W => 7,
            Handle::NW | Handle::SE => 8,
            Handle::NE | Handle::SW => 9,
        }
    }
}

pub const MIN: f64 = 8.0;

fn clamp_into(r: Rect, bw: f64, bh: f64) -> Rect {
    let w = (r.x1 - r.x0).min(bw);
    let h = (r.y1 - r.y0).min(bh);
    let x0 = r.x0.max(0.0).min(bw - w);
    let y0 = r.y0.max(0.0).min(bh - h);
    Rect::new(x0, y0, x0 + w, y0 + h)
}

/// Rectangle `r` after dragging `handle` by (dx, dy) image pixels inside an image of `bounds`.
pub fn drag_rect(r: Rect, handle: Handle, dx: f64, dy: f64, bounds: (f64, f64), ratio: Option<(u32, u32)>) -> Rect {
    let (bw, bh) = bounds;
    if handle == Handle::Move {
        return clamp_into(Rect::new(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy), bw, bh);
    }
    let (mut x0, mut y0, mut x1, mut y1) = (r.x0, r.y0, r.x1, r.y1);
    if handle.has('w') {
        x0 = (x1 - MIN).min((x0 + dx).max(0.0));
    }
    if handle.has('e') {
        x1 = (x0 + MIN).max((x1 + dx).min(bw));
    }
    if handle.has('n') {
        y0 = (y1 - MIN).min((y0 + dy).max(0.0));
    }
    if handle.has('s') {
        y1 = (y0 + MIN).max((y1 + dy).min(bh));
    }
    let Some(ratio) = ratio else { return Rect::new(x0, y0, x1, y1) };

    let k = ratio.0 as f64 / ratio.1 as f64;
    let anchor_x = if handle.has('w') {
        r.x1
    } else if handle.has('e') {
        r.x0
    } else {
        (r.x0 + r.x1) / 2.0
    };
    let anchor_y = if handle.has('n') {
        r.y1
    } else if handle.has('s') {
        r.y0
    } else {
        (r.y0 + r.y1) / 2.0
    };
    let mut w = x1 - x0;
    let mut h = y1 - y0;
    if handle.is_side() {
        // Edge handles drive one dimension; the other follows, centred on the opposite axis.
        if matches!(handle, Handle::E | Handle::W) {
            h = w / k;
        } else {
            w = h * k;
        }
    } else if w / h > k {
        w = h * k; // corner: keep the dimension that moved least, shrink the other
    } else {
        h = w / k;
    }
    // Largest size that still fits between the anchor and the image border.
    let max_w = if handle.has('w') {
        anchor_x
    } else if handle.has('e') {
        bw - anchor_x
    } else {
        2.0 * anchor_x.min(bw - anchor_x)
    };
    let max_h = if handle.has('n') {
        anchor_y
    } else if handle.has('s') {
        bh - anchor_y
    } else {
        2.0 * anchor_y.min(bh - anchor_y)
    };
    let s = 1f64.min(max_w / w).min(max_h / h);
    w *= s;
    h *= s;
    if handle.has('w') {
        (x0, x1) = (anchor_x - w, anchor_x);
    } else if handle.has('e') {
        (x0, x1) = (anchor_x, anchor_x + w);
    } else {
        (x0, x1) = (anchor_x - w / 2.0, anchor_x + w / 2.0);
    }
    if handle.has('n') {
        (y0, y1) = (anchor_y - h, anchor_y);
    } else if handle.has('s') {
        (y0, y1) = (anchor_y, anchor_y + h);
    } else {
        (y0, y1) = (anchor_y - h / 2.0, anchor_y + h / 2.0);
    }
    clamp_into(Rect::new(x0, y0, x1, y1), bw, bh)
}

pub fn same_rect(a: &Rect, b: &Rect, eps: f64) -> bool {
    (a.x0 - b.x0).abs() < eps && (a.y0 - b.y0).abs() < eps && (a.x1 - b.x1).abs() < eps && (a.y1 - b.y1).abs() < eps
}

/// The handle under a screen point, given the crop's screen rectangle; `None` outside it.
/// Handles are 14 px squares on the corners and edge midpoints; inside the rectangle is `Move`.
#[allow(clippy::too_many_arguments)]
pub fn hit_test(sx0: f64, sy0: f64, sx1: f64, sy1: f64, px: f64, py: f64, allow_resize: bool, allow_move: bool) -> Option<Handle> {
    let (mx, my) = ((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0);
    let near = |a: f64, b: f64| (a - b).abs() <= 8.0;
    if allow_resize {
        let spots = [
            (Handle::NW, sx0, sy0),
            (Handle::NE, sx1, sy0),
            (Handle::SE, sx1, sy1),
            (Handle::SW, sx0, sy1),
            (Handle::N, mx, sy0),
            (Handle::S, mx, sy1),
            (Handle::W, sx0, my),
            (Handle::E, sx1, my),
        ];
        if let Some((h, _, _)) = spots.iter().find(|(_, x, y)| near(px, *x) && near(py, *y)) {
            return Some(*h);
        }
    }
    (allow_move && px >= sx0 && px <= sx1 && py >= sy0 && py <= sy1).then_some(Handle::Move)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDS: (f64, f64) = (400.0, 300.0);
    fn base() -> Rect {
        Rect::new(100.0, 80.0, 300.0, 230.0)
    }
    fn ratio_of(r: &Rect) -> f64 {
        (r.x1 - r.x0) / (r.y1 - r.y0)
    }

    #[test]
    fn move_keeps_the_size_and_stays_inside_the_image() {
        let r = drag_rect(base(), Handle::Move, 500.0, -500.0, BOUNDS, None);
        assert!((r.x1 - r.x0 - 200.0).abs() < 1e-9 && (r.y1 - r.y0 - 150.0).abs() < 1e-9);
        assert!((r.x1 - 400.0).abs() < 1e-9 && r.y0.abs() < 1e-9);
    }

    #[test]
    fn free_resize_moves_only_the_dragged_edges_and_respects_the_minimum_size() {
        let r = drag_rect(base(), Handle::E, 50.0, 0.0, BOUNDS, None);
        assert_eq!(r, Rect::new(100.0, 80.0, 350.0, 230.0));
        let tiny = drag_rect(base(), Handle::W, 1000.0, 0.0, BOUNDS, None);
        assert!(tiny.x1 - tiny.x0 >= 8.0);
    }

    #[test]
    fn constrained_resize_keeps_the_ratio_for_every_handle() {
        let handles = [Handle::N, Handle::S, Handle::E, Handle::W, Handle::NE, Handle::NW, Handle::SE, Handle::SW];
        for ratio in [(1, 1), (16, 9), (3, 2)] {
            let start = drag_rect(base(), Handle::SE, 0.0, 0.0, BOUNDS, Some(ratio));
            for h in handles {
                let r = drag_rect(start, h, 23.0, -17.0, BOUNDS, Some(ratio));
                assert!((ratio_of(&r) - ratio.0 as f64 / ratio.1 as f64).abs() < 1e-6, "{h:?} {ratio:?}");
                assert!(r.x0 >= -1e-9 && r.y0 >= -1e-9 && r.x1 <= 400.0 + 1e-9 && r.y1 <= 300.0 + 1e-9);
            }
        }
    }

    #[test]
    fn never_leaves_the_image_even_with_a_huge_drag() {
        let r = drag_rect(base(), Handle::SE, 5000.0, 5000.0, BOUNDS, Some((4, 3)));
        assert!(r.x1 <= 400.0 && r.y1 <= 300.0);
        assert!(same_rect(&r, &r, 0.5));
    }

    #[test]
    fn hit_testing_prefers_handles_over_the_inside() {
        assert_eq!(hit_test(100.0, 100.0, 300.0, 250.0, 102.0, 98.0, true, true), Some(Handle::NW));
        assert_eq!(hit_test(100.0, 100.0, 300.0, 250.0, 200.0, 175.0, true, true), Some(Handle::Move));
        assert_eq!(hit_test(100.0, 100.0, 300.0, 250.0, 200.0, 175.0, true, false), None);
        assert_eq!(hit_test(100.0, 100.0, 300.0, 250.0, 20.0, 20.0, true, true), None);
    }
}
