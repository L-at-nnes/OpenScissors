//! Photoshop files (PSD) holding the picture and the cut-out as layers, for a program that edits layers: the mask
//! stays editable there, as a layer mask. Written after the published file format specification (8-bit RGB,
//! uncompressed channels).

use crate::error::{Error, Result};
use crate::types::{Image, Mask};

/// The largest side a PSD file can hold (larger needs the PSB variant).
pub const PSD_MAX_SIDE: usize = 30_000;

struct Layer<'a> {
    name: &'a str,
    hidden: bool,
    /// The layer mask (255 = shown), the size of the picture.
    mask: Option<&'a Mask>,
}

/// A PSD of `image` with two layers, bottom to top: "Photo" (the picture, hidden) and "Cut-out" (the picture with
/// `mask` as its layer mask). The flattened image, which programs without layers show, is the cut-out on
/// transparency.
pub fn encode_psd(image: &Image, mask: &Mask) -> Result<Vec<u8>> {
    let (w, h) = (image.w, image.h);
    if w > PSD_MAX_SIDE || h > PSD_MAX_SIDE {
        return Err(Error::msg(format!("A PSD file holds at most {PSD_MAX_SIDE} px per side; this picture is {w} x {h} px.")));
    }
    if (mask.w, mask.h) != (w, h) || image.data.len() != w * h * 3 {
        return Err(Error::msg("the mask and the picture differ in size"));
    }
    let planes: Vec<Vec<u8>> = (0..3).map(|c| image.data.chunks_exact(3).map(|p| p[c]).collect()).collect();
    let layers = [Layer { name: "Photo", hidden: true, mask: None }, Layer { name: "Cut-out", hidden: false, mask: Some(mask) }];
    let channel_len = (2 + w * h) as u32; // compression (raw), then the samples

    let mut out = Vec::with_capacity(64 + w * h * 12);
    // Header: signature, version 1, 6 reserved bytes, 4 channels (RGB and transparency), size, 8 bits, RGB mode.
    out.extend_from_slice(b"8BPS");
    put16(&mut out, 1);
    out.extend_from_slice(&[0; 6]);
    put16(&mut out, 4);
    put32(&mut out, h as u32);
    put32(&mut out, w as u32);
    put16(&mut out, 8);
    put16(&mut out, 3);
    put32(&mut out, 0); // colour mode data
    put32(&mut out, 0); // image resources

    // Layer info: the count (negative: the first extra channel of the flattened image is its transparency), the
    // records, then the channels of every layer in the same order.
    let mut info = Vec::new();
    put16(&mut info, (-(layers.len() as i16)) as u16);
    for layer in &layers {
        put32(&mut info, 0); // top
        put32(&mut info, 0); // left
        put32(&mut info, h as u32); // bottom
        put32(&mut info, w as u32); // right
        let ids: &[i16] = if layer.mask.is_some() { &[0, 1, 2, -2] } else { &[0, 1, 2] };
        put16(&mut info, ids.len() as u16);
        for &id in ids {
            put16(&mut info, id as u16);
            put32(&mut info, channel_len);
        }
        info.extend_from_slice(b"8BIMnorm");
        info.extend_from_slice(&[255, 0, if layer.hidden { 2 } else { 0 }, 0]); // opacity, clipping, flags, filler
        let mut extra = Vec::new();
        match layer.mask {
            Some(_) => {
                put32(&mut extra, 20);
                for v in [0, 0, h as u32, w as u32] {
                    put32(&mut extra, v);
                }
                extra.extend_from_slice(&[0, 0, 0, 0]); // hidden outside the rectangle, flags, padding
            }
            None => put32(&mut extra, 0),
        }
        put32(&mut extra, 0); // blending ranges
                              // The name: a Pascal string padded to a multiple of 4 bytes.
        let name = layer.name.as_bytes();
        extra.push(name.len() as u8);
        extra.extend_from_slice(name);
        while extra.len() % 4 != 0 {
            extra.push(0);
        }
        put32(&mut info, extra.len() as u32);
        info.extend_from_slice(&extra);
    }
    for layer in &layers {
        for plane in &planes {
            put16(&mut info, 0);
            info.extend_from_slice(plane);
        }
        if let Some(m) = layer.mask {
            put16(&mut info, 0);
            info.extend_from_slice(&m.data);
        }
    }
    if info.len() % 2 == 1 {
        info.push(0);
    }
    put32(&mut out, (4 + info.len() + 4) as u32); // layer and mask information
    put32(&mut out, info.len() as u32);
    out.extend_from_slice(&info);
    put32(&mut out, 0); // global layer mask

    // The flattened image: raw, channel after channel.
    put16(&mut out, 0);
    for plane in &planes {
        out.extend_from_slice(plane);
    }
    out.extend_from_slice(&mask.data);
    Ok(out)
}

fn put16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read16(b: &[u8], at: usize) -> u16 {
        u16::from_be_bytes([b[at], b[at + 1]])
    }

    fn read32(b: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn the_file_holds_two_layers_with_a_layer_mask_and_the_flattened_cut_out() {
        let (w, h) = (5usize, 3usize);
        let mut image = Image::new(w, h);
        image.data.iter_mut().enumerate().for_each(|(i, v)| *v = i as u8);
        let mut mask = Mask::new(w, h);
        mask.data[7] = 255;
        let psd = encode_psd(&image, &mask).unwrap();
        assert_eq!(&psd[..4], b"8BPS");
        assert_eq!((read16(&psd, 12), read32(&psd, 14), read32(&psd, 18)), (4, h as u32, w as u32));
        // Colour mode data and image resources are empty; then the layer and mask information.
        assert_eq!((read32(&psd, 26), read32(&psd, 30)), (0, 0));
        let section = read32(&psd, 34) as usize;
        let info = read32(&psd, 38) as usize;
        assert_eq!(section, 4 + info + 4);
        assert_eq!(read16(&psd, 42) as i16, -2, "two layers, transparency in the flattened image");
        // Walk the two records.
        let mut at = 44;
        let mut names = Vec::new();
        for expected_channels in [3usize, 4] {
            assert_eq!((read32(&psd, at + 8), read32(&psd, at + 12)), (h as u32, w as u32));
            let channels = read16(&psd, at + 16) as usize;
            assert_eq!(channels, expected_channels);
            at += 18 + channels * 6;
            assert_eq!(&psd[at..at + 8], b"8BIMnorm");
            let flags = psd[at + 10];
            at += 12;
            let extra = read32(&psd, at) as usize;
            let mask_len = read32(&psd, at + 4) as usize;
            let name_at = at + 4 + 4 + mask_len + 4;
            let len = psd[name_at] as usize;
            names.push((String::from_utf8(psd[name_at + 1..name_at + 1 + len].to_vec()).unwrap(), flags, mask_len));
            assert_eq!(extra % 4, 0);
            at += 4 + extra;
        }
        assert_eq!(names, [("Photo".to_string(), 2, 0), ("Cut-out".to_string(), 0, 20)], "the photo hidden, the cut-out with a mask");
        // The channel data of both layers fills the rest of the layer info.
        let data = (3 + 4) * (2 + w * h);
        assert!(42 + info - (at + data) <= 1, "then the channels, and one byte at most to make the length even");
        // The flattened image ends the file: raw, R G B then the transparency, which is the mask.
        let end = psd.len();
        assert_eq!(&psd[end - w * h..], &mask.data[..]);
        assert_eq!(read16(&psd, end - 4 * w * h - 2), 0, "raw");
        assert_eq!(psd[end - 4 * w * h], 0, "the first red sample");
        assert!(encode_psd(&Image::new(PSD_MAX_SIDE + 1, 1), &Mask::new(PSD_MAX_SIDE + 1, 1)).is_err());
    }
}
