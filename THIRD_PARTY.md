# Third-party components

Dependencies are resolved by Cargo, not vendored.

| Component | Use | License |
|---|---|---|
| Slint 1.18 (winit backend, software renderer) | interface | Slint Royalty-free License 2.0 (used here), or GPL-3.0, or commercial. Attribution: `AboutSlint` in the menu |
| winit, rfd, sys-locale | window, native dialogs, system language | Apache-2.0 / MIT |
| ort (ONNX Runtime bindings, ONNX Runtime) | model inference | MIT / Apache-2.0 |
| image, fast_image_resize | codecs, resampling | MIT / Apache-2.0 |
| rayon, serde, serde_json, thiserror, sha2, hex, base64, ureq | core utilities, checked downloads | MIT / Apache-2.0 |
| Microsoft DirectML runtime (Windows, optional GPU provider, shipped in the portable folder) | GPU inference | MIT |

## Models (downloaded by the user at runtime, never committed or bundled)

| Model | License | Source |
|---|---|---|
| BiRefNet lite, BiRefNet (Zheng et al.) | MIT | `onnx-community/BiRefNet_lite-ONNX`, `onnx-community/BiRefNet-ONNX` on Hugging Face |
| BiRefNet lite, half precision | MIT | `onnx-community/BiRefNet_lite-ONNX` (`model_fp16.onnx`) |
| IS-Net general use (DIS, Qin et al., ECCV 2022) | Apache-2.0 | `isnet-general-use.onnx`, rembg release `v0.0.0` |
| SAM 2.1 hiera tiny (Meta) | Apache-2.0 | `onnx-community/sam2.1-hiera-tiny-ONNX` |

Each catalogue entry pins the SHA-256 of the file; a download that does not match is deleted.

The built-in "Local file" entry only appears when `OPENSCISSORS_MODEL` is set (development).
