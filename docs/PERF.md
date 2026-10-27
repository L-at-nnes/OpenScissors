# Performance and accessibility audit (M6)

Measured on the development machine (Windows 11, 8 cores, GTX 1660 SUPER 6 GB), release build.
Reproduce: `cargo run --release -p openscissors-core --example perf_audit -- <models_dir> isnet-general-use`.

## Targets of the specification (section 9)

| Step | Target | Measured |
|---|---|---|
| Guided upsampling to full format, 12 Mpx | <= 500 ms | 447 ms |
| Graph-cut (reduced to 600 px), 12 Mpx | <= 800 ms | 632 ms |
| Hair matting, 12 Mpx | <= 1500 ms | 858 ms |
| Preview after a stroke, both views, 12 Mpx | <= 50 ms | 28 ms |
| Pan / zoom redraw of the result view, 12 Mpx | <= 50 ms | 20 ms |
| Full-resolution export with shadow, 12 Mpx | <= 3000 ms | 243 ms |
| Peak heap, 24 Mpx edit and export | <= 2 GB | 0.47 GB |
| IS-Net inference, 1024 px image | <= 1000 ms | 541 ms |

Not in the table: prediction end to end on a 12 Mpx photo with IS-Net takes about 1.1 s (inference plus
resizing and the guided upsampling above).

The memory figure counts heap bytes held at once (a counting allocator), not the process working set, which
also holds the web view and the ONNX runtime.

## Models on this machine (900 x 600 image)

| Model | CPU | GPU (DirectML) |
|---|---|---|
| IS-Net general use | 0.6 s | 0.85 s |
| BiRefNet lite | 5 to 6 s | out of video memory (6 GB), falls back to the CPU |
| BiRefNet lite, half precision | 7 to 8 s | 8 s |

DirectML brings **no speed-up here**: IS-Net is dominated by work that stays on the CPU, and BiRefNet's
attention layers either do not fit or run no faster than the CPU on this GPU. The GPU option therefore stays
off by default (*Processing: CPU*), is marked experimental, and is verified with a first inference when
chosen: "Automatic" falls back to the CPU when the GPU session cannot run, "GPU" reports the error. A card
with more memory and tensor cores may behave differently; CUDA (NVIDIA) is the more promising provider and is
not included because it needs the CUDA toolkit libraries (several GB).

## Memory of the models (CPU, 900 x 600 image)

ONNX Runtime keeps a memory arena that grows well beyond what the network needs. Measured peak working set:

| Model | Arena on (fast) | Arena off (default) | Time per image |
|---|---|---|---|
| IS-Net general use | 1.5 GB | 0.44 GB | 0.53 s on, 0.67 s off |
| BiRefNet lite | 11 GB | 4.7 GB | 5.2 s on, 7.1 s off |
| BiRefNet lite, half precision | n/a | 4.8 GB | 7.5 s |

The default is the lean mode (arena off); *Use more memory for speed* in the models panel turns the arena
on. A 4000 x 3000 photo predicted in the real application peaks at about 5.2 GB with BiRefNet lite. Before a
model is loaded the free memory (physical and commit limit) is checked against `ram_mb` of its catalogue
entry, and a model that cannot fit is refused with advice instead of ending the process. BiRefNet full has no
measured figure yet and is not checked.

## Memory limit and 8K photos

The application keeps to **25% of the physical memory** (8 GB on a 32 GB machine; the web view is not counted).

- **No model at start.** The application opens with 32 MB. The model loads with the first image: the one the
  user chose, otherwise the lightest on disk (IS-Net, 0.6 GB) that fits the limit. A heavier model is a choice.
- **Refusal instead of a crash.** A model whose figure (`ram_mb`) is above the limit is refused with advice
  ("Choose a lighter model"); *Use more memory for speed* is ignored when it would pass the limit. Before a
  prediction, process memory + model + image working set is checked against the limit.
- **Guided upsampling in bands** of 64 rows (4 rows of margin): the 33 Mpx working buffers of the whole-image
  version (about 1.3 GB) became a few MB. The result is the same within 2/255 (`banded_upsampling_matches_the_whole_image_filter`).

Measured in the real application, 7680 x 4320 JPEG (33 Mpx):

| Model | Peak working set | Time to the mask |
|---|---|---|
| IS-Net | 1.14 GB (the command line took 2.2 GB before the bands) | 2.2 s |
| BiRefNet lite | 5.6 GB, 0.5 GB at rest afterwards | 10.4 s |

`OPENSCISSORS_BUDGET_MB` replaces the 25% figure (to test a small machine): with 2000 MB the application
starts on IS-Net and refuses BiRefNet lite with the message above.

## Where the memory goes (whole process tree, default settings)

Measured with IS-Net, the default, on the first (web view) version of the interface; the Slint figures are in the next section.

| State | Application | Web view (6 processes) | Total (working set) |
|---|---|---|---|
| Started, nothing loaded | 31 MB | 355 MB | 386 MB |
| After a 1280 x 720 image | 237 MB | 384 MB | 621 MB |
| After an 8K photo | 396 MB | 396 MB | 792 MB |
| Smart select on top (SAM 2 tiny) | + 0.2 GB at rest, 1.0 to 1.3 GB at the peak | | |

What weighs, in order: the model chosen (IS-Net 0.6 GB, BiRefNet lite 5.2 GB: ten times more), the ONNX Runtime
memory arena (3.4 times more than needed, now off), the web view (a fixed 350 MB; its start-up options change
nothing measurable except disabling the GPU, which saves 100 MB but slows the views), and the pixels of the open
image. Nothing is loaded at start, and a model downloaded later is never activated by itself.

## Native interface (Slint)

The first interface was a web view (Tauri + Svelte); the figures above for "the whole process tree" were measured
with it. The interface was then rewritten with Slint (software renderer, no web view, no GPU context):

| State | Web view version (tree) | Slint version (one process) |
|---|---|---|
| Started, nothing loaded | 386 MB | 28 MB |
| After a 1280 x 720 image, IS-Net | 621 MB | 238 MB (peak 689 MB during the inference) |
| After an 8K photo, IS-Net | 792 MB | 428 MB (peak 1.33 GB during the inference) |

An empty Slint window costs 28 MB of working set (12 MB private); the OpenGL renderer of Slint (femtovg) costs 132 MB
for the same window, so the software renderer is the default (`--features gl` selects the other one).

## Accessibility

- **Contrast:** every text and component colour of the design tokens is checked against its surfaces
  (`crates/app/tests/contrast.rs`): primary, secondary and hint text at 4.5:1 or better on every surface, accent
  and tool colours at 4.5:1 on the panels.
- **Screen readers:** Slint exposes the controls through the platform accessibility API (AccessKit); buttons,
  sliders, switches, combo boxes and tabs carry a role and a label. Not tested with a real screen reader.
- **Keyboard:** every tool and action has a shortcut; the arrow keys pan the views; sliders take the arrow keys
  when focused. <kbd>Alt</kbd>+arrows move the crop (1 px, <kbd>Shift</kbd> 10 px); it is resized with the pointer
  (handles) and with the aspect ratio and Fit to result controls.
- **Not covered:** axe-core (a web tool) no longer applies; the native file dialogs belong to the operating system.
