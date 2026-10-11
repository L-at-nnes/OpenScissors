//! Viewport rendering off the interface thread: only the latest request per view is kept, so a
//! burst of pans costs one render, not one per event.

use std::sync::{Arc, Condvar, Mutex};

use openscissors_core::types::Rect;

use crate::state::AppState;
use crate::viewport::Region;

#[derive(Clone, Debug)]
pub struct Job {
    /// 0 = original, 1 = result
    pub side: usize,
    pub doc: u64,
    pub region: Region,
    pub w: usize,
    pub h: usize,
    pub hair: bool,
    /// Device position of the first pixel, so the transparency checker stays put while panning.
    pub origin: (i64, i64),
    pub cell: usize,
}

pub struct Output {
    pub job: Job,
    pub pixels: Vec<u8>,
}

type Slots = (Mutex<[Option<Job>; 2]>, Condvar);

pub struct Renderer {
    slots: Arc<Slots>,
}

impl Renderer {
    /// Start the render thread; `deliver` is called from it for every finished image.
    pub fn new(state: Arc<AppState>, deliver: impl Fn(Output) + Send + 'static) -> Self {
        let slots: Arc<Slots> = Arc::new((Mutex::new([None, None]), Condvar::new()));
        let worker = slots.clone();
        std::thread::Builder::new()
            .name("render".into())
            .spawn(move || loop {
                let job = {
                    let (lock, cv) = &*worker;
                    let mut g = lock.lock().expect("render slots");
                    loop {
                        if let Some(i) = (0..2).find(|&i| g[i].is_some()) {
                            break g[i].take().expect("slot");
                        }
                        g = cv.wait(g).expect("render wait");
                    }
                };
                if let Some(out) = render(&state, job) {
                    deliver(out);
                }
            })
            .expect("render thread");
        Self { slots }
    }

    pub fn submit(&self, job: Job) {
        let (lock, cv) = &*self.slots;
        let side = job.side;
        lock.lock().expect("render slots")[side] = Some(job);
        cv.notify_one();
    }
}

const MAX_PIXELS: usize = 24_000_000;

fn render(state: &AppState, job: Job) -> Option<Output> {
    if job.w == 0 || job.h == 0 || job.w * job.h > MAX_PIXELS {
        return None;
    }
    let region = Rect::new(job.region.x0, job.region.y0, job.region.x1, job.region.y1);
    let mut rgba = {
        let mut g = state.doc.lock().ok()?;
        let doc = g.as_mut()?;
        if doc.id != job.doc {
            return None;
        }
        if job.side == 0 {
            doc.render_source(region, job.w, job.h, job.hair)
        } else {
            doc.render_result(region, job.w, job.h)
        }
    };
    if job.side == 1 {
        checker(&mut rgba.data, &job);
    }
    Some(Output { job, pixels: rgba.data })
}

/// Composite straight-alpha pixels over a grey checker (the usual transparency cue).
fn checker(px: &mut [u8], job: &Job) {
    const LIGHT: [u32; 3] = [0x9a, 0x9a, 0x9c];
    const DARK: [u32; 3] = [0x7c, 0x7c, 0x80];
    let cell = job.cell.max(1) as i64;
    for y in 0..job.h {
        let cy = (job.origin.1 + y as i64).div_euclid(cell);
        for x in 0..job.w {
            let i = (y * job.w + x) * 4;
            let a = px[i + 3] as u32;
            if a == 255 {
                continue;
            }
            let cx = (job.origin.0 + x as i64).div_euclid(cell);
            let bg = if (cx + cy) & 1 == 0 { LIGHT } else { DARK };
            for c in 0..3 {
                px[i + c] = ((px[i + c] as u32 * a + bg[c] * (255 - a) + 127) / 255) as u8;
            }
            px[i + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(w: usize, h: usize) -> Job {
        Job { side: 1, doc: 1, region: Region { x0: 0.0, y0: 0.0, x1: 1.0, y1: 1.0 }, w, h, hair: false, origin: (0, 0), cell: 2 }
    }

    #[test]
    fn opaque_pixels_are_kept_and_transparent_ones_show_the_checker() {
        let mut px = vec![0u8; 4 * 4 * 4];
        px[0..4].copy_from_slice(&[10, 20, 30, 255]);
        checker(&mut px, &job(4, 4));
        assert_eq!(&px[0..4], &[10, 20, 30, 255]);
        assert_eq!(px[4 + 3], 255);
        // Cells of 2 px alternate: pixel (2, 0) has the other grey than pixel (1, 0).
        assert_ne!(px[4], px[8]);
    }

    #[test]
    fn half_transparent_pixels_blend_with_the_checker() {
        let mut px = vec![255, 255, 255, 128];
        checker(&mut px, &job(1, 1));
        assert!(px[0] > 0x9a && px[0] < 255 && px[3] == 255);
    }
}
