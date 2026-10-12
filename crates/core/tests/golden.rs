//! Golden test: needs a local IS-Net style ONNX model, e.g.
//! `OPENSCISSORS_MODEL=C:\path\model.onnx cargo test -p openscissors-core --test golden`

use std::path::{Path, PathBuf};

use openscissors_core::io::load_image;
use openscissors_core::model::{ModelSpec, OutputNorm, Predictor};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data").join(name)
}

fn isnet_spec() -> ModelSpec {
    ModelSpec {
        id: "isnet-local".into(),
        name: "IS-Net (local file)".into(),
        description: String::new(),
        url: String::new(),
        sha256: String::new(),
        size_bytes: 0,
        license: "Apache-2.0".into(),
        file_name: "model.onnx".into(),
        input_size: [1024, 1024],
        mean: [0.5, 0.5, 0.5],
        std: [1.0, 1.0, 1.0],
        input_name: None,
        output_name: None,
        output: OutputNorm::Minmax,
        recommended: false,
        kind: openscissors_core::model::ModelKind::Segmenter,
        files: vec![],
        ram_mb: 0,
        folder: None,
    }
}

#[test]
fn predict_matches_reference_mask() {
    let Ok(model) = std::env::var("OPENSCISSORS_MODEL") else {
        eprintln!("OPENSCISSORS_MODEL not set: skipped");
        return;
    };
    let predictor = Predictor::load(&isnet_spec(), Path::new(&model)).expect("load model");
    let img = load_image(&data("synthetic_figure.png")).unwrap();
    let reference = image::open(data("synthetic_figure_mask_reference.png")).unwrap().to_luma8();
    let t = std::time::Instant::now();
    let mask = predictor.predict(&img, true).unwrap();
    eprintln!("inference: {:?}", t.elapsed());
    assert_eq!((mask.w as u32, mask.h as u32), reference.dimensions());
    let mae = mask.data.iter().zip(reference.as_raw()).map(|(&a, &b)| (a as f64 - b as f64).abs()).sum::<f64>() / mask.data.len() as f64;
    eprintln!("MAE vs reference: {mae:.3}/255");
    assert!(mae <= 3.0, "MAE {mae}");
}
