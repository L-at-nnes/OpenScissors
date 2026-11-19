# Third-party components

Dependencies are resolved by Cargo, not vendored. Every licence in the dependency tree is checked in CI against the
permissive list of `deny.toml` (`cargo deny check licenses`); `cargo deny list` prints each crate with its licence.

| Component | Use | License |
|---|---|---|
| Slint 1.18 (winit backend, software renderer) | interface | Slint Royalty-free License 2.0 (used here), or GPL-3.0, or commercial. Attribution: `AboutSlint` in the menu |
| winit, rfd, sys-locale | window, native dialogs, system language | Apache-2.0 / MIT |
| ort (ONNX Runtime bindings, ONNX Runtime) | model inference | MIT / Apache-2.0 |
| image, fast_image_resize | codecs, resampling | MIT / Apache-2.0 |
| moxcms (also used by `image`) | colour profile conversion to sRGB | BSD-3-Clause / Apache-2.0 |
| webp, libwebp (C library, built from source) | lossy WebP export | MIT or Apache-2.0 (wrapper) / BSD-3-Clause (libwebp) |
| ravif, rav1e (through `image`) | AVIF export | BSD-3-Clause / BSD-2-Clause |
| tiff (also used by `image`) | compressed TIFF export with a resolution | MIT |
| jxl-oxide | JPEG XL import | MIT / Apache-2.0 |
| rayon, serde, serde_json, thiserror, sha2, hex, base64, ureq | core utilities, checked downloads | MIT / Apache-2.0 |
| rustls, ring, webpki-roots (through ureq) | HTTPS for the model downloads | Apache-2.0 / ISC / MIT; CDLA-Permissive-2.0 (root certificates) |
| arboard | system clipboard | MIT / Apache-2.0 |
| ICU4X, parley, fontique, zbus, accesskit (through Slint) | text layout, system integration, accessibility | Unicode-3.0 / MIT / Apache-2.0 |
| Microsoft DirectML runtime (Windows, optional GPU provider, shipped in the portable folder) | GPU inference | MIT |

## Models (downloaded by the user at runtime, never committed or bundled)

| Model | License | Source |
|---|---|---|
| BiRefNet lite, BiRefNet (Zheng et al.) | MIT | `onnx-community/BiRefNet_lite-ONNX`, `onnx-community/BiRefNet-ONNX` on Hugging Face |
| BiRefNet lite, half precision | MIT | `onnx-community/BiRefNet_lite-ONNX` (`model_fp16.onnx`) |
| BiRefNet portrait (Zheng et al.) | MIT | `onnx-community/BiRefNet-portrait-ONNX` |
| IS-Net general use (DIS, Qin et al., ECCV 2022) | Apache-2.0 | `isnet-general-use.onnx`, rembg release `v0.0.0` |
| U²-Net lite (U²-Net, Qin et al., 2020) | Apache-2.0 | `u2netp.onnx`, rembg release `v0.0.0` |
| IS-Net anime | Apache-2.0 | `isnet-anime.onnx`, rembg release `v0.0.0` |
| SAM 2.1 hiera tiny (Meta) | Apache-2.0 | `onnx-community/sam2.1-hiera-tiny-ONNX` |

Each catalogue entry pins the SHA-256 of the file; a download that does not match is deleted. The exceptions are
U²-Net lite and IS-Net anime: their release publishes no checksum, so they are downloaded unchecked and marked
"Not verified".

The built-in "Local file" entry only appears when `OPENSCISSORS_MODEL` is set (development).
