//! SAM 2 prompt segmentation ("Smart select"): click what to keep, what to leave out, or draw
//! a box. The image encoder runs once per image; the light decoder runs per prompt.
//!
//! Model: SAM 2.1 hiera tiny (`onnx-community/sam2.1-hiera-tiny-ONNX`), two networks.
//! - `vision_encoder`: `pixel_values` 1x3x1024x1024 (image stretched, `x/255`, ImageNet mean/std) gives
//!   `image_embeddings.0/.1/.2` of 1x32x256x256, 1x64x128x128 and 1x256x64x64, computed once per image.
//! - `prompt_encoder_mask_decoder`: `input_points` 1x1xNx2 (pixels scaled to the 1024 frame), `input_labels`
//!   1x1xN (int64, 1 keep / 0 leave out), `input_boxes` 1xBx4, and the three embeddings, gives `iou_scores`
//!   1x1x3 and `pred_masks` 1x1x3x256x256 (logits). The mask with the best score is used.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde::Deserialize;

use super::predict::{cpu_session, upsample_guided};
use crate::error::{Error, Result};
use crate::imaging::resize::{resize_image, Interp};
use crate::types::{Image, Mask};

/// Network frame: images are stretched to this square.
const FRAME: usize = 1024;
/// Side of the decoder's low-resolution masks.
const MASK_SIDE: usize = 256;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// More prompts than this make no sense for one object.
const MAX_PROMPTS: usize = 64;

/// A user prompt, in image pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Prompt {
    /// A click: `keep = true` marks the object, `false` marks what to leave out.
    Point { x: f32, y: f32, keep: bool },
    /// A rectangle around the object.
    Box { x0: f32, y0: f32, x1: f32, y1: f32 },
}

/// Output of the image encoder for one image.
pub struct Embeddings {
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
}

pub struct Sam2 {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
}

/// The decoder inputs for `prompts` on an image of `w` x `h`: points (`x, y` pairs), their
/// labels (1 keep, 0 leave out) and boxes (`x0, y0, x1, y1`), all in the 1024 frame.
pub fn prompt_tensors(prompts: &[Prompt], w: usize, h: usize) -> Result<(Vec<f32>, Vec<i64>, Vec<f32>)> {
    if prompts.is_empty() {
        return Err(Error::msg("give at least one point or box"));
    }
    if prompts.len() > MAX_PROMPTS {
        return Err(Error::msg("too many prompts"));
    }
    let (sx, sy) = (FRAME as f32 / w as f32, FRAME as f32 / h as f32);
    let (mut points, mut labels, mut boxes) = (Vec::new(), Vec::new(), Vec::new());
    for p in prompts {
        match *p {
            Prompt::Point { x, y, keep } => {
                points.extend_from_slice(&[(x * sx).clamp(0.0, FRAME as f32), (y * sy).clamp(0.0, FRAME as f32)]);
                labels.push(i64::from(keep));
            }
            Prompt::Box { x0, y0, x1, y1 } => {
                let (xa, xb) = (x0.min(x1), x0.max(x1));
                let (ya, yb) = (y0.min(y1), y0.max(y1));
                boxes.extend_from_slice(&[xa * sx, ya * sy, xb * sx, yb * sy].map(|v| v.clamp(0.0, FRAME as f32)));
            }
        }
    }
    Ok((points, labels, boxes))
}

/// Mask of the best of the decoder's candidates: `logits` holds `candidates` masks of
/// `MASK_SIDE` squared logits; the one with the highest score becomes a 0..255 mask.
pub fn best_mask(scores: &[f32], logits: &[f32]) -> Result<Mask> {
    let plane = MASK_SIDE * MASK_SIDE;
    if scores.is_empty() || logits.len() < scores.len() * plane {
        return Err(Error::msg("unexpected decoder output"));
    }
    let best = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
    let data = logits[best * plane..(best + 1) * plane].par_iter().map(|&v| ((1.0 / (1.0 + (-v).exp())) * 255.0).round() as u8).collect();
    Ok(Mask { w: MASK_SIDE, h: MASK_SIDE, data })
}

impl Sam2 {
    /// Load from a folder holding `vision_encoder.onnx` and `prompt_encoder_mask_decoder.onnx`
    /// (with their `.onnx_data` side files).
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Self {
            encoder: Mutex::new(cpu_session(&dir.join("vision_encoder.onnx"), false)?),
            decoder: Mutex::new(cpu_session(&dir.join("prompt_encoder_mask_decoder.onnx"), false)?),
        })
    }

    /// Run the image encoder (about a second on a CPU). Keep the result for every prompt.
    pub fn embed(&self, img: &Image) -> Result<Embeddings> {
        let small = resize_image(img, FRAME, FRAME, Interp::Bilinear);
        let plane = FRAME * FRAME;
        let mut x = vec![0f32; 3 * plane];
        {
            let (r, rest) = x.split_at_mut(plane);
            let (g, b) = rest.split_at_mut(plane);
            r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).zip(small.data.par_chunks_exact(3)).for_each(
                |(((r, g), b), p)| {
                    *r = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
                    *g = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
                    *b = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
                },
            );
        }
        let tensor = Tensor::from_array(([1usize, 3, FRAME, FRAME], x))?;
        let mut session = self.encoder.lock().map_err(|_| Error::msg("SAM 2 encoder poisoned"))?;
        let out = session.run(ort::inputs!["pixel_values" => tensor])?;
        let take = |i: usize| -> Result<Vec<f32>> { Ok(out[i].try_extract_tensor::<f32>()?.1.to_vec()) };
        Ok(Embeddings { e0: take(0)?, e1: take(1)?, e2: take(2)? })
    }

    /// Segment with `prompts`. The result has the size of `img` (255 = selected), with edges
    /// refined on the image like the AI prediction is.
    pub fn select(&self, emb: &Embeddings, prompts: &[Prompt], img: &Image) -> Result<Mask> {
        let (points, labels, boxes) = prompt_tensors(prompts, img.w, img.h)?;
        let (np, nb) = (labels.len(), boxes.len() / 4);
        let mut session = self.decoder.lock().map_err(|_| Error::msg("SAM 2 decoder poisoned"))?;
        let out = session.run(ort::inputs![
            "input_points" => Tensor::from_array(([1usize, 1, np, 2], points))?,
            "input_labels" => Tensor::from_array(([1usize, 1, np], labels))?,
            "input_boxes" => Tensor::from_array(([1usize, nb, 4], boxes))?,
            "image_embeddings.0" => Tensor::from_array(([1usize, 32, 256, 256], emb.e0.clone()))?,
            "image_embeddings.1" => Tensor::from_array(([1usize, 64, 128, 128], emb.e1.clone()))?,
            "image_embeddings.2" => Tensor::from_array(([1usize, 256, 64, 64], emb.e2.clone()))?,
        ])?;
        let scores = out["iou_scores"].try_extract_tensor::<f32>()?.1.to_vec();
        let low = best_mask(&scores, out["pred_masks"].try_extract_tensor::<f32>()?.1)?;
        Ok(upsample_guided(img, &low))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_scaled_to_the_network_frame() {
        let prompts = [
            Prompt::Point { x: 450.0, y: 300.0, keep: true },
            Prompt::Point { x: 0.0, y: 600.0, keep: false },
            Prompt::Box { x0: 600.0, y0: 400.0, x1: 300.0, y1: 100.0 }, // corners in any order
        ];
        let (points, labels, boxes) = prompt_tensors(&prompts, 900, 600).unwrap();
        assert_eq!(labels, vec![1, 0]);
        let near = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3);
        assert!(near(&points, &[512.0, 512.0 * 300.0 / 300.0 * 1.0, 0.0, 1024.0]), "{points:?}");
        assert!(
            near(&boxes, &[300.0 * 1024.0 / 900.0, 100.0 * 1024.0 / 600.0, 600.0 * 1024.0 / 900.0, 400.0 * 1024.0 / 600.0]),
            "{boxes:?}"
        );
    }

    #[test]
    fn prompts_are_validated_and_clamped() {
        assert!(prompt_tensors(&[], 10, 10).is_err());
        let many = vec![Prompt::Point { x: 1.0, y: 1.0, keep: true }; MAX_PROMPTS + 1];
        assert!(prompt_tensors(&many, 10, 10).is_err());
        let (p, _, _) = prompt_tensors(&[Prompt::Point { x: -50.0, y: 5000.0, keep: true }], 100, 100).unwrap();
        assert_eq!(p, vec![0.0, 1024.0]);
    }

    #[test]
    fn the_candidate_with_the_best_score_becomes_a_probability_mask() {
        let plane = MASK_SIDE * MASK_SIDE;
        let mut logits = vec![-10.0f32; 3 * plane];
        logits[plane..2 * plane].iter_mut().for_each(|v| *v = 10.0); // candidate 1 is "all selected"
        let m = best_mask(&[0.2, 0.9, 0.5], &logits).unwrap();
        assert_eq!((m.w, m.h), (MASK_SIDE, MASK_SIDE));
        assert!(m.data.iter().all(|&v| v == 255));
        let m = best_mask(&[0.9, 0.2, 0.5], &logits).unwrap();
        assert!(m.data.iter().all(|&v| v == 0));
        assert!(best_mask(&[0.5], &logits[..10]).is_err());
    }
}
