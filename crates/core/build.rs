//! macOS only: link the clang runtime library.
//!
//! The ONNX Runtime binary for macOS contains the Core ML provider, whose Objective-C++ code checks the version of the
//! system with `__isPlatformVersionAtLeast`. That function lives in the clang runtime, and Rust links with
//! `-nodefaultlibs`, so the link fails with an undefined symbol unless the library is named here. The library is asked
//! of the C compiler that does the linking, so it is the one of the installed Xcode or command line tools.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CC");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let printed = Command::new(cc).arg("--print-file-name=libclang_rt.osx.a").output().ok().filter(|o| o.status.success());
    let printed = printed.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    // A compiler that does not know the library prints just its name back: then there is nothing to link.
    let library = Path::new(&printed);
    match library.parent().filter(|_| library.is_absolute() && library.is_file()) {
        Some(dir) => {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
        }
        None => println!("cargo:warning=libclang_rt.osx.a was not found; linking the Core ML provider may fail"),
    }
}
