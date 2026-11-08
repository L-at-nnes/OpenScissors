//! One-click corrections of a whole mask: what an AI prediction typically needs (a hole in the subject, specks of
//! background kept, a soft edge that should be crisp, the opposite of what was wanted).

use std::collections::VecDeque;

use serde::Deserialize;

use crate::types::Mask;

/// Mask values above this are the subject (the threshold of the rest of the pipeline).
const SUBJECT: u8 = 127;
/// Values above this are visible (see `render::crop::BBOX_MIN_ALPHA`): a speck is anything visible.
const VISIBLE: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskTool {
    /// Fill the holes inside the subject: background areas that do not reach the edge of the picture.
    FillHoles,
    /// Remove the small islands of subject and fill the small holes in it (specks and pinholes).
    RemoveSpecks,
    /// Keep only the largest part of the subject.
    KeepLargest,
    /// Swap subject and background.
    Invert,
    /// Make the soft parts of the mask crisp.
    Harden,
}

/// `mask` corrected by `tool`.
pub fn apply(tool: MaskTool, mask: &Mask) -> Mask {
    match tool {
        MaskTool::FillHoles => fill_holes(mask),
        MaskTool::RemoveSpecks => remove_specks(mask),
        MaskTool::KeepLargest => keep_largest(mask),
        MaskTool::Invert => Mask { w: mask.w, h: mask.h, data: mask.data.iter().map(|&v| 255 - v).collect() },
        MaskTool::Harden => harden(mask),
    }
}

/// Background (values up to `SUBJECT`) that cannot reach the edge of the picture through background becomes subject.
fn fill_holes(m: &Mask) -> Mask {
    let outside = reach_from_edges(m, |v| v <= SUBJECT);
    let data = m.data.iter().zip(&outside).map(|(&v, &out)| if v <= SUBJECT && !out { 255 } else { v }).collect();
    Mask { w: m.w, h: m.h, data }
}

/// Which pixels satisfying `inside` are connected (4-neighbours) to the edge of the picture through such pixels.
fn reach_from_edges(m: &Mask, inside: impl Fn(u8) -> bool) -> Vec<bool> {
    let (w, h) = (m.w, m.h);
    let mut seen = vec![false; w * h];
    let mut queue = VecDeque::new();
    let push = |i: usize, seen: &mut Vec<bool>, queue: &mut VecDeque<usize>| {
        if !seen[i] && inside(m.data[i]) {
            seen[i] = true;
            queue.push_back(i);
        }
    };
    for x in 0..w {
        push(x, &mut seen, &mut queue);
        push((h - 1) * w + x, &mut seen, &mut queue);
    }
    for y in 0..h {
        push(y * w, &mut seen, &mut queue);
        push(y * w + w - 1, &mut seen, &mut queue);
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        if x > 0 {
            push(i - 1, &mut seen, &mut queue);
        }
        if x + 1 < w {
            push(i + 1, &mut seen, &mut queue);
        }
        if y > 0 {
            push(i - w, &mut seen, &mut queue);
        }
        if y + 1 < h {
            push(i + w, &mut seen, &mut queue);
        }
    }
    seen
}

/// Label the 8-connected regions of the pixels satisfying `inside`: the label of each pixel (0 = outside) and the
/// size of each label (index 0 unused).
fn regions(m: &Mask, inside: impl Fn(u8) -> bool) -> (Vec<u32>, Vec<usize>) {
    let (w, h) = (m.w, m.h);
    let mut label = vec![0u32; w * h];
    let mut sizes = vec![0usize];
    let mut queue = VecDeque::new();
    for start in 0..w * h {
        if label[start] != 0 || !inside(m.data[start]) {
            continue;
        }
        let id = sizes.len() as u32;
        sizes.push(0);
        label[start] = id;
        queue.push_back(start);
        while let Some(i) = queue.pop_front() {
            sizes[id as usize] += 1;
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x + dx, y + dy);
                    if (dx, dy) == (0, 0) || nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if label[j] == 0 && inside(m.data[j]) {
                        label[j] = id;
                        queue.push_back(j);
                    }
                }
            }
        }
    }
    (label, sizes)
}

/// Keep the visible regions of the subject for which `keep(size, largest)` holds; the others are cleared.
fn keep_regions(m: &Mask, keep: impl Fn(usize, usize) -> bool) -> Mask {
    let (label, sizes) = regions(m, |v| v > VISIBLE);
    let largest = sizes.iter().copied().max().unwrap_or(0);
    let kept: Vec<bool> = sizes.iter().map(|&s| keep(s, largest)).collect();
    let data = m.data.iter().zip(&label).map(|(&v, &l)| if l != 0 && !kept[l as usize] { 0 } else { v }).collect();
    Mask { w: m.w, h: m.h, data }
}

fn keep_largest(m: &Mask) -> Mask {
    let (_, sizes) = regions(m, |v| v > VISIBLE);
    let Some(best) = sizes.iter().enumerate().skip(1).max_by_key(|(_, &s)| s).map(|(i, _)| i) else { return m.clone() };
    let (label, _) = regions(m, |v| v > VISIBLE);
    let data = m.data.iter().zip(&label).map(|(&v, &l)| if l != 0 && l as usize != best { 0 } else { v }).collect();
    Mask { w: m.w, h: m.h, data }
}

/// A speck is a region smaller than 1 % of the largest one and than 0.1 % of the picture (at least 16 pixels count).
fn is_speck(size: usize, largest: usize, picture: usize) -> bool {
    size < (largest / 100).min(picture / 1000).max(16)
}

fn remove_specks(m: &Mask) -> Mask {
    let picture = m.w * m.h;
    let islands_gone = keep_regions(m, |size, largest| !is_speck(size, largest, picture));
    // Pinholes: small background regions enclosed by the subject.
    let outside = reach_from_edges(&islands_gone, |v| v <= SUBJECT);
    let (label, sizes) = regions(&islands_gone, |v| v <= SUBJECT);
    let largest_subject = regions(&islands_gone, |v| v > SUBJECT).1.into_iter().max().unwrap_or(0);
    let data = islands_gone
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let l = label[i] as usize;
            if l != 0 && !outside[i] && is_speck(sizes[l], largest_subject, picture) {
                255
            } else {
                v
            }
        })
        .collect();
    Mask { w: m.w, h: m.h, data }
}

/// Stretch the values around the middle four times: a soft edge of 64 values becomes one of 16, flat areas stay.
fn harden(m: &Mask) -> Mask {
    let data = m.data.iter().map(|&v| ((v as f32 - 127.5) * 4.0 + 127.5).round().clamp(0.0, 255.0) as u8).collect();
    Mask { w: m.w, h: m.h, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 40 x 30 subject with a 6 x 6 hole, a 2 x 2 pinhole, a 3 x 3 speck outside, and a second, smaller part.
    fn sample() -> Mask {
        let (w, h) = (100, 60);
        let mut m = Mask::new(w, h);
        let mut fill = |x0: usize, y0: usize, x1: usize, y1: usize, v: u8| {
            for y in y0..y1 {
                for x in x0..x1 {
                    m.data[y * w + x] = v;
                }
            }
        };
        fill(10, 10, 50, 40, 255);
        fill(20, 20, 26, 26, 0); // hole
        fill(40, 30, 42, 32, 0); // pinhole
        fill(80, 5, 83, 8, 255); // speck
        fill(60, 30, 75, 50, 255); // second part
        m
    }

    fn at(m: &Mask, x: usize, y: usize) -> u8 {
        m.data[y * m.w + x]
    }

    #[test]
    fn holes_are_filled_and_the_background_that_reaches_the_edge_stays() {
        let out = apply(MaskTool::FillHoles, &sample());
        assert_eq!((at(&out, 22, 22), at(&out, 40, 30)), (255, 255), "hole and pinhole filled");
        assert_eq!((at(&out, 5, 5), at(&out, 55, 20)), (0, 0), "outside stays background");
    }

    #[test]
    fn specks_and_pinholes_go_but_real_parts_and_holes_stay() {
        let out = apply(MaskTool::RemoveSpecks, &sample());
        assert_eq!(at(&out, 81, 6), 0, "the speck is gone");
        assert_eq!(at(&out, 40, 30), 255, "the pinhole is filled");
        assert_eq!(at(&out, 22, 22), 0, "a real hole (36 px of a 6000 px picture... and 3% of the subject) stays");
        assert_eq!(at(&out, 65, 40), 255, "the second part is not a speck");
    }

    #[test]
    fn only_the_largest_part_is_kept() {
        let out = apply(MaskTool::KeepLargest, &sample());
        assert_eq!((at(&out, 30, 15), at(&out, 65, 40), at(&out, 81, 6)), (255, 0, 0));
        assert_eq!(apply(MaskTool::KeepLargest, &Mask::new(4, 4)), Mask::new(4, 4), "nothing to keep");
    }

    #[test]
    fn invert_and_harden() {
        let m = Mask { w: 4, h: 1, data: vec![0, 100, 160, 255] };
        assert_eq!(apply(MaskTool::Invert, &m).data, vec![255, 155, 95, 0]);
        assert_eq!(apply(MaskTool::Harden, &m).data, vec![0, 18, 255, 255]);
    }
}
