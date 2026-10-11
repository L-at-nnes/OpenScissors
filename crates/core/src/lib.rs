//! OpenScissors core: imaging, AI prediction, masks, document model and rendering.
//! No GUI dependency; the Tauri app only calls into this crate.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_memcpy, clippy::chunks_exact_to_as_chunks)]

pub mod batch;
pub mod document;
pub mod error;
pub mod imaging;
pub mod io;
pub mod mask;
pub mod model;
pub mod params;
pub mod plugin;
pub mod project;
pub mod render;
pub mod settings;
pub mod sysmem;
pub mod types;

pub use error::{Error, Result};
pub use types::{Image, Image16, Mask, Rect, Rgba, Rgba16};
