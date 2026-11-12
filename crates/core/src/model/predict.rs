//! AI mask prediction (spec section 4.2).

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::registry::{ModelSpec, OutputNorm};
use crate::error::{Error, Result};
use crate::imaging::guided::guided_filter;
use crate::imaging::resize::{resize_image, resize_mask, Interp};
use crate::sync::LockExt;
use crate::types::{Image, Mask};

const UPSAMPLE_R: usize = 1;
const UPSAMPLE_EPS: f32 = 162.5625; // (0.05 * 255)^2

/// Where to run the network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accelerator {
    /// The GPU when the platform supports it and it works, otherwise the CPU.
    Auto,
    /// The processor: slower on big models, but always works and is the reference behaviour.
    #[default]
    Cpu,
    /// The GPU; loading fails when it is not available.
    Gpu,
}

pub struct Predictor {
    spec: ModelSpec,
    session: Mutex<Session>,
    input: String,
    backend: &'static str,
}

/// The ONNX Runtime environment of the process. ONNX Runtime allows one per process, so it is created on first use
/// and kept for the whole run; every session is built from it. Telemetry is turned off explicitly, whatever the build
/// of ONNX Runtime would do by default.
pub(crate) fn environment() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("openscissors").with_telemetry(false).build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::Ort(e.clone()))
}

/// `fast_memory` keeps ONNX Runtime's memory arena on: quicker (about a third for BiRefNet) but
/// the process then holds more than twice the memory (11 GB instead of 4.7 GB for BiRefNet lite).
pub(crate) fn cpu_session(model_path: &Path, fast_memory: bool) -> Result<Session> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut b = Session::builder(environment()?)?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_intra_threads(threads)?;
    if !fast_memory {
        b = b.with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])?.with_memory_pattern(false)?;
    }
    Ok(b.commit_from_file(model_path)?)
}

/// A graphics card provider of ONNX Runtime, in the order they are tried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuProvider {
    /// NVIDIA cards, in a build with the `cuda` feature (needs the CUDA 12 runtime and cuDNN 9 installed).
    #[cfg(feature = "cuda")]
    Cuda,
    /// Any DirectX 12 card on Windows.
    #[cfg(windows)]
    DirectMl,
    /// The GPU and the Neural Engine of a Mac.
    #[cfg(target_os = "macos")]
    CoreMl,
}

impl GpuProvider {
    /// What the status shows when a model runs on it.
    fn label(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            GpuProvider::Cuda => "GPU (CUDA)",
            #[cfg(windows)]
            GpuProvider::DirectMl => "GPU (DirectML)",
            #[cfg(target_os = "macos")]
            GpuProvider::CoreMl => "GPU (Core ML)",
        }
    }
}

/// The providers this build can try, best first.
fn gpu_providers() -> Vec<GpuProvider> {
    #[allow(unused_mut, clippy::vec_init_then_push)] // the pushes depend on the platform and the features
    let mut list = Vec::new();
    #[cfg(feature = "cuda")]
    list.push(GpuProvider::Cuda);
    #[cfg(windows)]
    list.push(GpuProvider::DirectMl);
    #[cfg(target_os = "macos")]
    list.push(GpuProvider::CoreMl);
    list
}

/// A session on the first graphics card provider that loads `model_path`, and its label. Each provider is tried on its
/// own, so that a CUDA build without the CUDA runtime still falls back on DirectML.
fn gpu_session(model_path: &Path) -> Result<(Session, &'static str)> {
    let mut last = Error::msg("GPU acceleration is not available on this platform yet");
    for provider in gpu_providers() {
        match provider_session(provider, model_path) {
            Ok(session) => return Ok((session, provider.label())),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn provider_session(provider: GpuProvider, model_path: &Path) -> Result<Session> {
    use ort::ep;
    let builder = Session::builder(environment()?)?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    let mut builder = match provider {
        #[cfg(feature = "cuda")]
        GpuProvider::Cuda => builder.with_execution_providers([ep::CUDA::default().build().error_on_failure()])?,
        #[cfg(windows)]
        GpuProvider::DirectMl => builder
            .with_execution_providers([ep::DirectML::default().build().error_on_failure()])?
            // Both are required by the DirectML provider.
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?,
        #[cfg(target_os = "macos")]
        GpuProvider::CoreMl => builder.with_execution_providers([ep::CoreML::default().build().error_on_failure()])?,
    };
    Ok(builder.commit_from_file(model_path)?)
}

impl Predictor {
    /// Load `spec` from `model_path` on the CPU (deterministic; used by tests and tools).
    pub fn load(spec: &ModelSpec, model_path: &Path) -> Result<Self> {
        Self::load_with(spec, model_path, Accelerator::Cpu)
    }

    /// Like [`Predictor::load_with`], choosing speed over memory when `fast_memory` is set.
    pub fn load_with_memory(spec: &ModelSpec, model_path: &Path, accelerator: Accelerator, fast_memory: bool) -> Result<Self> {
        Self::load_inner(spec, model_path, accelerator, fast_memory)
    }

    /// Load `spec` from `model_path` on the requested accelerator. A GPU session is checked with
    /// a first inference: some graphs load fine and then fail to run (out of video memory).
    /// With `Auto` such a failure falls back to the CPU; with `Gpu` it is reported.
    pub fn load_with(spec: &ModelSpec, model_path: &Path, accelerator: Accelerator) -> Result<Self> {
        Self::load_inner(spec, model_path, accelerator, false)
    }

    fn load_inner(spec: &ModelSpec, model_path: &Path, accelerator: Accelerator, fast_memory: bool) -> Result<Self> {
        match accelerator {
            Accelerator::Cpu => Self::from_session(spec, cpu_session(model_path, fast_memory)?, "CPU"),
            Accelerator::Gpu => {
                let (session, label) = gpu_session(model_path)?;
                let p = Self::from_session(spec, session, label)?;
                p.warmup()?;
                Ok(p)
            }
            Accelerator::Auto => {
                let gpu = gpu_session(model_path).and_then(|(s, label)| Self::from_session(spec, s, label));
                match gpu.and_then(|p| p.warmup().map(|_| p)) {
                    Ok(p) => Ok(p),
                    Err(_) => Self::from_session(spec, cpu_session(model_path, fast_memory)?, "CPU"),
                }
            }
        }
    }

    fn from_session(spec: &ModelSpec, session: Session, backend: &'static str) -> Result<Self> {
        let input = match &spec.input_name {
            Some(n) => n.clone(),
            None => session.inputs().first().ok_or_else(|| Error::msg("model has no input"))?.name().to_string(),
        };
        Ok(Self { spec: spec.clone(), session: Mutex::new(session), input, backend })
    }

    /// "GPU" or "CPU": where the network really runs.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn warmup(&self) -> Result<()> {
        let [w, h] = self.spec.input_size;
        self.run(&Image::filled(w, h, [128, 128, 128])).map(|_| ())
    }

    /// Predict a foreground mask with the size of `img` (255 = foreground).
    /// `compat = true` stretches the image to the network size (ratio not preserved);
    /// `false` letterboxes it.
    pub fn predict(&self, img: &Image, compat: bool) -> Result<Mask> {
        let [in_w, in_h] = self.spec.input_size;
        let (w, h) = (img.w, img.h);
        let s = in_w.max(in_h) as f64 / w.max(h) as f64;
        let sw = ((w as f64 * s).round() as usize).max(1);
        let sh = ((h as f64 * s).round() as usize).max(1);
        let small = resize_image(img, sw, sh, if s < 1.0 { Interp::Area } else { Interp::Bicubic });

        let raw = if compat {
            self.run(&resize_image(&small, in_w, in_h, Interp::Bilinear))?
        } else {
            let mut canvas = Image::filled(in_w, in_h, [128, 128, 128]);
            let (cw, ch) = (sw.min(in_w), sh.min(in_h));
            for y in 0..ch {
                canvas.data[y * in_w * 3..(y * in_w + cw) * 3].copy_from_slice(&small.data[y * sw * 3..(y * sw + cw) * 3]);
            }
            let full = self.run(&canvas)?;
            let mut crop = Mask::new(cw, ch);
            for y in 0..ch {
                crop.data[y * cw..(y + 1) * cw].copy_from_slice(&full.data[y * in_w..y * in_w + cw]);
            }
            crop
        };
        let y = resize_mask(&raw, sw, sh, Interp::Bilinear);
        if (sw, sh) == (w, h) {
            return Ok(y);
        }
        Ok(upsample_guided(img, &y))
    }

    /// Normalise `img` (already the network size), infer, return the output as a 0..255 mask.
    fn run(&self, img: &Image) -> Result<Mask> {
        let [in_w, in_h] = self.spec.input_size;
        debug_assert_eq!((img.w, img.h), (in_w, in_h));
        let plane = in_w * in_h;
        let mut x = vec![0f32; 3 * plane];
        let (mean, std) = (self.spec.mean, self.spec.std);
        {
            let (r, rest) = x.split_at_mut(plane);
            let (g, b) = rest.split_at_mut(plane);
            r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).zip(img.data.par_chunks_exact(3)).for_each(|(((r, g), b), p)| {
                *r = (p[0] as f32 / 255.0 - mean[0]) / std[0];
                *g = (p[1] as f32 / 255.0 - mean[1]) / std[1];
                *b = (p[2] as f32 / 255.0 - mean[2]) / std[2];
            });
        }
        let tensor = Tensor::from_array(([1usize, 3, in_h, in_w], x))?;
        let mut session = self.session.guard();
        let outputs = session.run(ort::inputs![self.input.as_str() => tensor])?;
        let out = match &self.spec.output_name {
            Some(n) => outputs.get(n.as_str()).ok_or_else(|| Error::msg(format!("model has no output '{n}'")))?,
            None => &outputs[0],
        };
        let (shape, data) = out.try_extract_tensor::<f32>()?;
        if data.len() < plane {
            return Err(Error::msg(format!("unexpected output shape {shape:?}")));
        }
        // Output is [.., H, W]; take the last plane (batch/channel 0).
        let y = &data[data.len() - plane..];
        Ok(Mask { w: in_w, h: in_h, data: to_u8(y, self.spec.output) })
    }
}

fn to_u8(y: &[f32], norm: OutputNorm) -> Vec<u8> {
    match norm {
        OutputNorm::Minmax => {
            let (mn, mx) = y.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let range = (mx - mn).max(1e-8);
            y.par_iter().map(|&v| ((v - mn) / range * 255.0) as u8).collect()
        }
        OutputNorm::Sigmoid => y.par_iter().map(|&v| ((1.0 / (1.0 + (-v).exp())) * 255.0).clamp(0.0, 255.0) as u8).collect(),
        OutputNorm::None => y.par_iter().map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8).collect(),
    }
}

/// Rows filtered together by [`upsample_guided`]: working buffers stay a few MB however big the image is.
const BAND_ROWS: usize = 64;
/// Rows a band reads beyond its own: the filter looks 2 * `UPSAMPLE_R` rows away, so the rows kept are exact.
const BAND_MARGIN: usize = 4;

/// Edge-aware upsampling of a lower-resolution mask to the size of `img`.
/// Works band by band, so an 8K photo needs megabytes of working memory instead of gigabytes.
pub fn upsample_guided(img: &Image, low: &Mask) -> Mask {
    let (w, h) = (img.w, img.h);
    let xs = bilinear_taps(low.w, w);
    let ys = bilinear_taps(low.h, h);
    let mut data = vec![0u8; w * h];
    data.par_chunks_mut(BAND_ROWS * w).enumerate().for_each(|(band, out)| {
        let y0 = band * BAND_ROWS;
        let rows = out.len() / w;
        let (a0, a1) = (y0.saturating_sub(BAND_MARGIN), (y0 + rows + BAND_MARGIN).min(h));
        let n = a1 - a0;
        let mut guide = vec![0f32; w * n];
        let mut src = vec![0f32; w * n];
        for (i, y) in (a0..a1).enumerate() {
            for (x, g) in guide[i * w..(i + 1) * w].iter_mut().enumerate() {
                let p = &img.data[(y * w + x) * 3..(y * w + x) * 3 + 3];
                *g = ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000) as f32;
            }
            let (ya, yb, fy) = ys[y];
            let (ra, rb) = (&low.data[ya * low.w..(ya + 1) * low.w], &low.data[yb * low.w..(yb + 1) * low.w]);
            for (s, &(xa, xb, fx)) in src[i * w..(i + 1) * w].iter_mut().zip(&xs) {
                let top = ra[xa] as f32 * (1.0 - fx) + ra[xb] as f32 * fx;
                let bottom = rb[xa] as f32 * (1.0 - fx) + rb[xb] as f32 * fx;
                *s = top * (1.0 - fy) + bottom * fy;
            }
        }
        let filtered = guided_filter(&guide, &src, w, n, UPSAMPLE_R, UPSAMPLE_EPS, 1);
        for (o, &v) in out.iter_mut().zip(&filtered[(y0 - a0) * w..(y0 - a0 + rows) * w]) {
            *o = v.clamp(0.0, 255.0) as u8;
        }
    });
    Mask { w, h, data }
}

/// For each of `dst` positions the two neighbours in a row of `src` samples and the weight of the second
/// (bilinear, pixel centres aligned).
fn bilinear_taps(src: usize, dst: usize) -> Vec<(usize, usize, f32)> {
    (0..dst)
        .map(|i| {
            let f = ((i as f32 + 0.5) * src as f32 / dst as f32 - 0.5).clamp(0.0, (src - 1) as f32);
            let a = f.floor() as usize;
            (a, (a + 1).min(src - 1), f - a as f32)
        })
        .collect()
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    fn the_providers_of_the_build_are_tried_in_order_and_named() {
        let list = gpu_providers();
        #[cfg(windows)]
        assert_eq!(list.last().map(|p| p.label()), Some("GPU (DirectML)"), "DirectML last on Windows: {list:?}");
        #[cfg(feature = "cuda")]
        assert_eq!(list.first().map(|p| p.label()), Some("GPU (CUDA)"), "CUDA first when built in");
        #[cfg(target_os = "macos")]
        assert_eq!(list, vec![GpuProvider::CoreMl]);
        #[cfg(all(not(windows), not(target_os = "macos"), not(feature = "cuda")))]
        assert!(list.is_empty() && gpu_session(Path::new("x.onnx")).is_err(), "no GPU provider in this build");
        assert!(list.iter().all(|p| p.label().starts_with("GPU (")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_normalisations() {
        assert_eq!(to_u8(&[2.0, 4.0, 6.0], OutputNorm::Minmax), vec![0, 127, 255]);
        let s = to_u8(&[-20.0, 0.0, 20.0], OutputNorm::Sigmoid);
        assert_eq!((s[0], s[1], s[2]), (0, 127, 255));
        assert_eq!(to_u8(&[0.0, 0.5, 1.0], OutputNorm::None), vec![0, 127, 255]);
    }

    #[test]
    fn banded_upsampling_matches_the_whole_image_filter() {
        // 203 x 157: not a multiple of the band height, so the last band is short.
        let (w, h) = (203usize, 157usize);
        let mut img = Image::new(w, h);
        let mut seed = 12345u32;
        for v in img.data.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed >> 24) as u8;
        }
        let mut low = Mask::new(51, 40);
        for (i, v) in low.data.iter_mut().enumerate() {
            *v = if (i % 51 + i / 51) % 7 < 3 { 255 } else { 0 };
        }
        let guide: Vec<f32> = img.gray().data.iter().map(|&v| v as f32).collect();
        let src: Vec<f32> = resize_mask(&low, w, h, Interp::Bilinear).data.iter().map(|&v| v as f32).collect();
        let whole = guided_filter(&guide, &src, w, h, UPSAMPLE_R, UPSAMPLE_EPS, 1);
        let banded = upsample_guided(&img, &low);
        let worst = whole.iter().zip(&banded.data).map(|(a, &b)| (a.clamp(0.0, 255.0) - b as f32).abs()).fold(0f32, f32::max);
        assert!(worst <= 2.0, "worst difference {worst}");
    }

    #[test]
    fn guided_upsampling_preserves_flat_regions_and_edge_order() {
        // Left half dark, right half bright; the low-res mask is blurry around the true edge.
        let (w, h) = (64usize, 64usize);
        let mut img = Image::filled(w, h, [20, 20, 20]);
        for y in 0..h {
            for x in 32..w {
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[230, 230, 230]);
            }
        }
        let mut low = Mask::new(16, 16);
        for y in 0..16 {
            for x in 8..16 {
                low.data[y * 16 + x] = 255;
            }
        }
        let guided = upsample_guided(&img, &low);
        assert_eq!((guided.w, guided.h), (w, h));
        assert!(guided.get(2, 10) < 5 && guided.get(61, 10) > 250); // far from the edge: unchanged
        assert!(guided.get(30, 10) < guided.get(34, 10)); // edge stays monotonic
    }
}
