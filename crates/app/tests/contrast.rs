//! WCAG contrast of the design tokens of `ui/theme.slint` against the surfaces they are drawn on.

use std::collections::HashMap;

fn tokens() -> HashMap<String, [f64; 3]> {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/ui/theme.slint")).expect("theme.slint");
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("out property <color> ") else { continue };
        let Some((name, value)) = rest.split_once(':') else { continue };
        let hex = value.trim().trim_end_matches(';').trim_start_matches('#');
        if hex.len() == 6 {
            let n = u32::from_str_radix(hex, 16).expect("hex colour");
            out.insert(name.trim().to_string(), [(n >> 16 & 255) as f64, (n >> 8 & 255) as f64, (n & 255) as f64]);
        }
    }
    out
}

fn luminance(c: [f64; 3]) -> f64 {
    let lin = |v: f64| {
        let v = v / 255.0;
        if v <= 0.03928 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])
}

fn ratio(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (la, lb) = (luminance(a), luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

#[test]
fn text_and_component_colours_reach_4_5_to_1_on_every_surface() {
    let t = tokens();
    for surface in ["bg", "canvas", "surface-1", "surface-2", "surface-3"] {
        for fg in ["text", "text-2", "text-3"] {
            let r = ratio(t[fg], t[surface]);
            assert!(r >= 4.5, "{fg} on {surface}: {r:.2}:1");
        }
    }
    for surface in ["surface-1", "surface-2"] {
        for fg in ["accent", "keep", "remove", "hair", "warn", "danger"] {
            let r = ratio(t[fg], t[surface]);
            assert!(r >= 4.5, "{fg} on {surface}: {r:.2}:1");
        }
    }
    // Text on the accent fill of primary buttons.
    assert!(ratio(t["accent-ink"], t["accent"]) >= 4.5);
    assert!(ratio(t["accent-ink"], t["accent-strong"]) >= 4.5);
}
