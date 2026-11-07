//! 3 x 3 matrices (colour covariances of GrabCut and of the matting windows).

/// Determinant of `c`.
pub(crate) fn det3(c: &[[f64; 3]; 3]) -> f64 {
    c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
        + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
}

/// Inverse of `c`, whose determinant is `det` (not zero).
pub(crate) fn inv3(c: &[[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let d = 1.0 / det;
    [
        [
            (c[1][1] * c[2][2] - c[1][2] * c[2][1]) * d,
            (c[0][2] * c[2][1] - c[0][1] * c[2][2]) * d,
            (c[0][1] * c[1][2] - c[0][2] * c[1][1]) * d,
        ],
        [
            (c[1][2] * c[2][0] - c[1][0] * c[2][2]) * d,
            (c[0][0] * c[2][2] - c[0][2] * c[2][0]) * d,
            (c[0][2] * c[1][0] - c[0][0] * c[1][2]) * d,
        ],
        [
            (c[1][0] * c[2][1] - c[1][1] * c[2][0]) * d,
            (c[0][1] * c[2][0] - c[0][0] * c[2][1]) * d,
            (c[0][0] * c[1][1] - c[0][1] * c[1][0]) * d,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_times_its_inverse_is_the_identity() {
        let c = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let inv = inv3(&c, det3(&c));
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| c[i][k] * inv[k][j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "{i},{j}: {v}");
            }
        }
        assert_eq!(det3(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]), 24.0);
    }
}
