//! Mask composition, simple mode: prediction, then keep/erase strokes in order. The boundary offset and the
//! feathering come later, when the result is rendered (`render::finished_alpha`).

use super::strokes::{paint_stroke, Stroke, StrokeKind};
use crate::types::Mask;

/// Apply one stroke to `alpha` in place (strokes of other kinds are ignored).
pub fn apply_stroke(alpha: &mut Mask, stroke: &Stroke) {
    if matches!(stroke.kind, StrokeKind::Keep | StrokeKind::Erase) {
        paint_stroke(alpha, stroke, stroke.kind.value());
    }
}

pub fn compose_alpha(predicted: Option<&Mask>, strokes: &[Stroke], w: usize, h: usize) -> Mask {
    let mut alpha = predicted.cloned().unwrap_or_else(|| Mask::new(w, h));
    for s in strokes {
        apply_stroke(&mut alpha, s);
    }
    alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stroke(kind: StrokeKind, x: f32, y: f32, w: u32) -> Stroke {
        Stroke { kind, width: w, points: vec![[x, y]] }
    }

    #[test]
    fn keep_and_erase_on_prediction() {
        let mut pred = Mask::new(50, 40);
        for y in 0..40 {
            for x in 25..50 {
                pred.data[y * 50 + x] = 200;
            }
        }
        let strokes = vec![stroke(StrokeKind::Keep, 5.5, 5.5, 5), stroke(StrokeKind::Erase, 40.5, 20.5, 5)];
        let a = compose_alpha(Some(&pred), &strokes, 50, 40);
        assert_eq!(a.get(5, 5), 255);
        assert_eq!(a.get(40, 20), 0);
        assert_eq!(a.get(49, 20), 200);
        assert_eq!(a.get(10, 30), 0);
    }

    #[test]
    fn no_prediction_starts_empty() {
        assert!(compose_alpha(None, &[], 4, 4).is_empty_mask());
    }

    /// The offset and the feathering of the result, as the renderer applies them.
    fn finish_alpha(alpha: &Mask, offset: i32, feather: u32) -> Mask {
        let mut p = crate::params::Params::default();
        p.mask.offset = offset;
        p.foreground.feather = feather;
        crate::render::finished_alpha(alpha, &p)
    }

    #[test]
    fn offset_zero_and_feather_zero_is_identity() {
        let mut m = Mask::new(20, 20);
        m.data[5 * 20 + 5] = 255;
        assert_eq!(finish_alpha(&m, 0, 0), m);
    }

    #[test]
    fn positive_offset_grows_negative_shrinks_feather_softens() {
        let mut m = Mask::new(40, 40);
        for y in 10..30 {
            for x in 10..30 {
                m.data[y * 40 + x] = 255;
            }
        }
        let area = |m: &Mask| m.data.iter().filter(|&&v| v > 127).count();
        assert!(area(&finish_alpha(&m, 3, 0)) > area(&m));
        assert!(area(&finish_alpha(&m, -3, 0)) < area(&m));
        let soft = finish_alpha(&m, 0, 2);
        assert!(soft.get(10, 20) > 0 && soft.get(10, 20) < 255);
    }
}
