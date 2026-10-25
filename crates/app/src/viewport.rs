//! Zoom and pan shared by both views: `screen = image * scale + (tx, ty)`, in logical pixels.

pub const MIN_SCALE: f64 = 0.02;
pub const MAX_SCALE: f64 = 64.0;
pub const ZOOM_STEP: f64 = 1.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub scale: f64,
    pub tx: f64,
    pub ty: f64,
    /// Follows the pane size (re-fitted when it changes) until the user zooms or pans.
    pub fit: bool,
}

impl Default for View {
    fn default() -> Self {
        Self { scale: 1.0, tx: 0.0, ty: 0.0, fit: true }
    }
}

/// A region in image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

fn clamp_scale(s: f64) -> f64 {
    s.clamp(MIN_SCALE, MAX_SCALE)
}

/// Largest scale that shows the whole image, centred, with a little air around it.
pub fn fit_view(img: (f64, f64), boxed: (f64, f64), margin: f64) -> View {
    let aw = (boxed.0 - margin * 2.0).max(1.0);
    let ah = (boxed.1 - margin * 2.0).max(1.0);
    let scale = clamp_scale((aw / img.0).min(ah / img.1));
    View { scale, tx: (boxed.0 - img.0 * scale) / 2.0, ty: (boxed.1 - img.1 * scale) / 2.0, fit: true }
}

/// Zoom by `factor` keeping the image point under (px, py) fixed.
pub fn zoom_at(v: View, factor: f64, px: f64, py: f64) -> View {
    let scale = clamp_scale(v.scale * factor);
    if scale == v.scale {
        return v;
    }
    let (ix, iy) = ((px - v.tx) / v.scale, (py - v.ty) / v.scale);
    View { scale, tx: px - ix * scale, ty: py - iy * scale, fit: false }
}

pub fn pan_by(v: View, dx: f64, dy: f64) -> View {
    View { tx: v.tx + dx, ty: v.ty + dy, fit: false, ..v }
}

pub fn to_image(v: &View, sx: f64, sy: f64) -> (f64, f64) {
    ((sx - v.tx) / v.scale, (sy - v.ty) / v.scale)
}

pub fn to_screen(v: &View, ix: f64, iy: f64) -> (f64, f64) {
    (ix * v.scale + v.tx, iy * v.scale + v.ty)
}

/// Part of the image visible in a pane, in image pixels, or `None` when nothing is.
pub fn visible_region(v: &View, img: (f64, f64), boxed: (f64, f64)) -> Option<Region> {
    visible_in(v, Region { x0: 0.0, y0: 0.0, x1: img.0, y1: img.1 }, boxed)
}

/// Part of `area` (image pixels) visible in a pane, or `None` when nothing is. The result view uses an area larger
/// than the image when the crop extends the canvas.
pub fn visible_in(v: &View, area: Region, boxed: (f64, f64)) -> Option<Region> {
    let x0 = (-v.tx / v.scale).max(area.x0);
    let y0 = (-v.ty / v.scale).max(area.y0);
    let x1 = area.x1.min((boxed.0 - v.tx) / v.scale);
    let y1 = area.y1.min((boxed.1 - v.ty) / v.scale);
    (x1 - x0 > 1e-6 && y1 - y0 > 1e-6).then_some(Region { x0, y0, x1, y1 })
}

/// Pixel grid for fetching `r`: size in device pixels, and the region that exactly matches it.
/// Edges are snapped to whole device pixels so the bitmap lands exactly where the region starts.
pub fn device_grid(v: &View, r: &Region, dpr: f64) -> (usize, usize, Region) {
    let x = ((v.tx + r.x0 * v.scale) * dpr).round();
    let y = ((v.ty + r.y0 * v.scale) * dpr).round();
    let x2 = ((v.tx + r.x1 * v.scale) * dpr).round();
    let y2 = ((v.ty + r.y1 * v.scale) * dpr).round();
    let w = (x2 - x).max(1.0);
    let h = (y2 - y).max(1.0);
    let region = Region {
        x0: (x / dpr - v.tx) / v.scale,
        y0: (y / dpr - v.ty) / v.scale,
        x1: (x2 / dpr - v.tx) / v.scale,
        y1: (y2 / dpr - v.ty) / v.scale,
    };
    (w as usize, h as usize, region)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_centres_the_image_and_keeps_it_inside_the_box() {
        let v = fit_view((400.0, 200.0), (500.0, 500.0), 0.0);
        assert!((v.scale - 1.25).abs() < 1e-9 && v.tx.abs() < 1e-9 && (v.ty - 125.0).abs() < 1e-9 && v.fit);
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor_fixed() {
        let v = View { scale: 1.0, tx: 10.0, ty: 20.0, fit: true };
        let before = to_image(&v, 200.0, 150.0);
        let z = zoom_at(v, 2.0, 200.0, 150.0);
        let after = to_image(&z, 200.0, 150.0);
        assert!((before.0 - after.0).abs() < 1e-9 && (before.1 - after.1).abs() < 1e-9 && !z.fit);
    }

    #[test]
    fn screen_and_image_coordinates_round_trip() {
        let v = View { scale: 2.5, tx: -30.0, ty: 12.0, fit: false };
        let (sx, sy) = to_screen(&v, 17.0, 9.0);
        let (ix, iy) = to_image(&v, sx, sy);
        assert!((ix - 17.0).abs() < 1e-9 && (iy - 9.0).abs() < 1e-9);
    }

    #[test]
    fn visible_region_is_clipped_to_the_image_and_none_when_off_screen() {
        let img = (100.0, 100.0);
        let r = visible_region(&View { scale: 2.0, tx: -50.0, ty: 0.0, fit: false }, img, (100.0, 300.0)).unwrap();
        assert_eq!(r, Region { x0: 25.0, y0: 0.0, x1: 75.0, y1: 100.0 });
        assert!(visible_region(&View { scale: 1.0, tx: 500.0, ty: 0.0, fit: false }, img, (100.0, 100.0)).is_none());
        assert!(visible_region(&pan_by(View::default(), -200.0, 0.0), img, (100.0, 100.0)).is_none());
    }

    #[test]
    fn the_visible_part_of_a_larger_area_reaches_past_the_image() {
        let area = Region { x0: -50.0, y0: 0.0, x1: 150.0, y1: 100.0 };
        let v = View { scale: 1.0, tx: 60.0, ty: 0.0, fit: false };
        assert_eq!(visible_in(&v, area, (300.0, 100.0)).unwrap(), Region { x0: -50.0, y0: 0.0, x1: 150.0, y1: 100.0 });
        assert_eq!(visible_region(&v, (100.0, 100.0), (300.0, 100.0)).unwrap(), Region { x0: 0.0, y0: 0.0, x1: 100.0, y1: 100.0 });
        // Only the extra margin is in the pane: the image itself is out of view.
        assert!(visible_region(&v, (100.0, 100.0), (30.0, 100.0)).is_none());
        assert_eq!(visible_in(&v, area, (30.0, 100.0)).unwrap(), Region { x0: -50.0, y0: 0.0, x1: -30.0, y1: 100.0 });
    }

    #[test]
    fn device_grid_snaps_to_whole_pixels_and_its_region_maps_back_exactly() {
        let v = View { scale: 0.37, tx: 13.3, ty: 5.9, fit: false };
        let r = visible_region(&v, (1000.0, 800.0), (400.0, 300.0)).unwrap();
        let (w, _h, region) = device_grid(&v, &r, 2.0);
        let back = (region.x1 - region.x0) * v.scale * 2.0;
        assert!((back - w as f64).abs() < 1e-6);
    }
}
