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

/// Calculates e^x
pub fn exp<T: From<f64> + Copy>(x: f64) -> T {
    let x_f64: f64 = Into::<f64>::into(x);
    let result_f64 = x_f64.exp();
    T::from(result_f64)
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

    #[test]
    fn test_exp() {
        assert!(numbers::approx_equal(exp::<f64>(1.0), 2.71828182846, 0.000000000001));
        assert!(numbers::approx_equal(exp::<f64>(2.0), 7.38905609893065, 0.000000000001));
        assert!(numbers::approx_equal(exp::<f64>(2.5), 12.182493960703473, 0.000000000001));
        assert!(numbers::approx_equal(exp::<f64>(5.0), 148.4131591025766, 0.000000000001));
        assert!(numbers::approx_equal(exp::<f64>(-5.0), 0.006737946999085467, 0.000000000001));
        assert!(numbers::approx_equal(exp::<f64>(0.0), 1.0, 0.000000000001));
    }
}
