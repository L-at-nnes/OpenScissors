# Contributing

Thanks for looking. A few things that make a change easy to accept:

- **Keep it small and tested.** `cargo test --workspace` and `cargo clippy --workspace --all-targets` must pass; `cargo fmt --all` before you commit.
- **The core stays free of any GUI crate.** If it can be tested without a window, it lives in `crates/core`.
- **Light by default.** A feature that costs memory or start-up time should be opt-in.
- **Strings.** Add the English text in the code (`t("...")` in Rust, `@tr("...")` in Slint) and its translation in every file of `crates/app/locales/`. The test tells you what is missing.
- **Look at it.** For interface changes, run the selftest (`cargo run --release -p openscissors --example selftest -- photo.jpg out/ tour`) and look at the snapshots, at the minimum window size too (`OSC_SIZE=960x600`) and in a long language (`OSC_LANG=de` or `ru`).
- **No weights, no foreign assets.** Models are downloaded at run time (URL and SHA-256 in `crates/core/src/model/registry.rs`); never commit `.onnx` files, icons or text taken from other software.

Commit messages: `type(scope): short description` (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`).

By contributing you agree that your work is released under the Apache-2.0 license of the project.
