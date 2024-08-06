use crate::combinatorics;

/// Approximates the sine of an angle using the Taylor series expansion
pub fn sin<T: Into<f64> + Copy>(x: T) -> f64 {
    let x = x.into();
    let mut result = 0.0;
    for n in 0..16 {
        let sign: f64 = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * x.powi(2 * n + 1) / combinatorics::factorial_i128((2 * n + 1).into()) as f64;
        result += term;
    }
    result
}

/// Approximates the cosine of an angle using the Taylor series expansion
pub fn cos<T: Into<f64> + Copy>(x: T) -> f64 {
    let x = x.into();
    let mut result: f64 = 0.0;
    for n in 0..16 {
        let sign: f64 = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * x.powi(2 * n) / combinatorics::factorial_i128((2 * n).into()) as f64;
        result += term;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers;

    #[test]
    fn test_sin() {
        assert!(numbers::approx_equal(sin(std::f64::consts::PI), 0.0, 0.0000001));
        assert!(numbers::approx_equal(sin(std::f64::consts::PI / 2.0), 1.0, 0.0000001));
        assert!(numbers::approx_equal(sin(std::f64::consts::PI / 4.0), 0.70710678119, 0.0000001));
        assert!(numbers::approx_equal(sin(std::f64::consts::PI * 2.0), 0.0, 0.0000001));
    }

    #[test]
    fn test_cos() {
        assert!(numbers::approx_equal(cos(std::f64::consts::PI), -1.0, 0.0000001));
        assert!(numbers::approx_equal(cos(std::f64::consts::PI / 2.0), 0.0, 0.0000001));
        assert!(numbers::approx_equal(cos(std::f64::consts::PI / 4.0), 0.70710678119, 0.0000001));
        assert!(numbers::approx_equal(cos(std::f64::consts::PI * 2.0), 1.0, 0.0000001));
    }
}
