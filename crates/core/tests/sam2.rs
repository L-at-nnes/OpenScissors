//! SAM 2 with the real model. Needs a folder with the four ONNX files:
//! `OPENSCISSORS_SAM_DIR=<folder> cargo test -p openscissors-core --test sam2 -- --nocapture`

use std::path::{Path, PathBuf};

use openscissors_core::io::load_image;
use openscissors_core::model::{Prompt, Sam2};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data").join(name)
}

#[test]
fn points_and_boxes_select_the_subject() {
    let Ok(dir) = std::env::var("OPENSCISSORS_SAM_DIR") else {
        eprintln!("OPENSCISSORS_SAM_DIR not set: skipped");
        return;
    };
    let sam = Sam2::load(Path::new(&dir)).expect("load SAM 2");
    let img = load_image(&data("synthetic_figure.png")).unwrap(); // 900 x 600, figure around x = 450
    let t = std::time::Instant::now();
    let emb = sam.embed(&img).expect("encode");
    eprintln!("encoder: {:?}", t.elapsed());

    let area = |m: &openscissors_core::types::Mask| m.data.iter().filter(|&&v| v > 127).count() as f64 / m.data.len() as f64;

    // One click on the yellow body.
    let t = std::time::Instant::now();
    let m = sam.select(&emb, &[Prompt::Point { x: 450.0, y: 450.0, keep: true }], &img).unwrap();
    eprintln!("decoder + refinement: {:?}, selected {:.1}%", t.elapsed(), area(&m) * 100.0);
    assert_eq!((m.w, m.h), (img.w, img.h));
    assert_eq!(m.get(450, 450), 255, "the clicked pixel is selected");
    assert_eq!(m.get(60, 60), 0, "the corner is not");
    assert!(area(&m) > 0.02 && area(&m) < 0.45);

    // A box around head and body selects both.
    let b = sam.select(&emb, &[Prompt::Box { x0: 320.0, y0: 110.0, x1: 580.0, y1: 570.0 }], &img).unwrap();
    eprintln!("box selected {:.1}%", area(&b) * 100.0);
    assert!(b.get(450, 240) > 127, "head");
    assert!(b.get(450, 450) > 127, "body");
    assert_eq!(b.get(60, 60), 0);
    assert!(area(&b) > area(&m) * 0.9);

    // A keep click plus a leave-out click on the head separates the two parts.
    let both = sam
        .select(&emb, &[Prompt::Point { x: 450.0, y: 450.0, keep: true }, Prompt::Point { x: 450.0, y: 240.0, keep: false }], &img)
        .unwrap();
    assert!(both.get(450, 450) > 127);
    assert!(both.get(450, 240) < 127, "the head is left out");

    // No prompt is an error.
    assert!(sam.select(&emb, &[], &img).is_err());
}
