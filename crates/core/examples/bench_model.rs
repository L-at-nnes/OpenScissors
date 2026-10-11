//! `cargo run --release -p openscissors-core --example bench_model -- <models_dir> <model_id> <image> [cpu|gpu|auto]`
use std::path::Path;
use std::time::Instant;

use openscissors_core::io::load_image;
use openscissors_core::model::{Accelerator, ModelRegistry, Predictor};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reg = ModelRegistry::new(&a[1]);
    let spec = reg.get(&a[2]).expect("model id");
    let img = load_image(Path::new(&a[3])).expect("image");
    let t = Instant::now();
    let accel = match a.get(4).map(String::as_str) {
        Some("gpu") => Accelerator::Gpu,
        Some("auto") => Accelerator::Auto,
        _ => Accelerator::Cpu,
    };
    let p = Predictor::load_with(&spec, &reg.model_path(&spec), accel).expect("load");
    println!("backend: {}, load: {:?}", p.backend(), t.elapsed());
    let t = Instant::now();
    p.warmup().expect("warmup");
    println!("warmup: {:?}", t.elapsed());
    for i in 0..3 {
        let t = Instant::now();
        let m = p.predict(&img, true).expect("predict");
        println!("predict #{i}: {:?} ({}x{})", t.elapsed(), m.w, m.h);
    }
}
