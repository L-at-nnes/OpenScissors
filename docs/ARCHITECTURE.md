# Architecture

Three crates, one rule: **`crates/core` knows nothing about windows.** Everything that can be tested without a screen is tested there.

```
crates/core   imaging, models, masks, document, rendering, export, settings
crates/app    the window (Slint), the controller that connects it to the core
crates/cli    command line: models, predict, cut, batch
```

## Core

| Module | What it does |
|---|---|
| `model/` | `ModelRegistry` (catalogue + user models, SHA-256 checked downloads), `Predictor` (ONNX Runtime session, normalisation, guided up-sampling in bands), `Sam2` (encoder once per image, decoder per prompt) |
| `mask/` | own GrabCut and Boykov–Kolmogorov max-flow, closed-form matting (coarse to fine), strokes |
| `document.rs` | the open image: strokes, parameters, undo stack, the mask pipeline (a provisional mask is shown at once, the exact one is computed in the background), 16-bit export |
| `render/` | viewport rendering at screen resolution, crop, filters, shadow, inpainting. The same code renders the export at scale 1 |
| `project.rs`, `batch.rs`, `plugin.rs`, `settings.rs`, `sysmem.rs` | `.osproj` / `.psm` files, batch, plug-ins (opt-in, with a timeout), settings, memory budget |

## App

```
ui/*.slint     theme tokens, widgets, dock, panels, dialogs, main window (Backend global = the contract with Rust)
src/ctl.rs     Ctl: interface state + pushing it into the Slint properties (coalesced refreshes)
src/ops.rs     document operations: open, save, export, projects, marks, parameters, batch, plug-ins
src/models.rs  memory limits, model loading, prediction, background mask jobs, downloads, smart select
src/input.rs   pointer and keyboard
src/render.rs  render thread: keeps the latest request per view
src/dialogs.rs questions drawn by the interface, system file choosers
src/window.rs  remembers position and size of the window
src/state.rs   portable folders, shared state
src/i18n.rs    translations (keys are the English strings); .slint files use @tr, compiled from the same JSON
```

**Threads.** The interface thread only reads state and sets properties. Everything slow (decoding, inference, export, downloads) runs on worker threads and reports back through `Ctl::refresh`, which coalesces requests and runs on the interface thread. Rendering has its own thread with a "latest request wins" slot per view, so a burst of pans costs one render.

**Pixels.** The window never holds the full image. It shows the last rendered region at the position the current pan and zoom give it (a Slint binding), and replaces it when the render thread delivers the next one.

**Memory.** The budget (25 % of the RAM, 50 % optional) is checked before a model loads and before a prediction. ONNX Runtime's arena is off by default (it costs ~2.3x more); big images are filtered in bands.

## Testing the interface without a mouse

`crates/app/examples/selftest.rs` opens the real window, plays a scenario (real pointer, key and wheel events, or JSON steps) and saves a snapshot after each step. The screenshots in the README come from it.

## Translations

`crates/app/locales/*.json` map the English string to its translation. A test checks that every string used by the code exists in every language, that placeholders and shortcuts survive, and that nothing is left over.
