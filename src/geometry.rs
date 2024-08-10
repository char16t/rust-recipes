use crate::combinatorics;

/// Approximates the sine of an angle using the Taylor series expansion
pub fn sin<T: Into<f64> + Copy>(x: T) -> f64 {
    let x = x.into();
    let mut result: f64 = 0.0;
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
    let result_f64: f64 = x_f64.exp();
    T::from(result_f64)
}

#[derive(Debug, Clone, Copy)]
pub struct Complex<T> {
    real: T,
    imaginary: T
}
impl<T> Complex<T>
where 
    f64: From<T>,
    T: Copy + std::convert::From<f64> + std::ops::Mul<Output = T> + std::ops::Neg<Output = T>
{
    pub fn new(real: T, imaginary: T) -> Self {
        Self { real, imaginary }
    }
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            real: r * cos(theta).into(),
            imaginary: r * sin(theta).into()
        }
    }
    pub fn re(&self) -> T {
        self.real
    }
    pub fn im(&self) -> T {
        self.imaginary
    }
    /// Calculates e^(self)
    pub fn exp(&self) -> Self {
        let re: T = exp::<T>(self.real.into()) * (cos(self.imaginary).into());
        let im: T = exp::<T>(self.real.into()) * (sin(self.imaginary).into());

        Complex::new(re, im)
    }
    /// Complex conjugate
    pub fn conj(&self) -> Complex<T> {
        Self { real: self.real, imaginary: -self.imaginary } 
    }
}
impl<T> std::ops::Neg for Complex<T>
where
    T: std::ops::Neg<Output = T>
{
    type Output = Complex<T>;

    fn neg(self) -> Complex<T> {
        Complex {
            real: -self.real,
            imaginary: -self.imaginary,
        }
    }
}

impl<T> std::ops::Add for Complex<T>
where
    T: std::ops::Add<Output = T>
{
    type Output = Complex<T>;

    fn add(self, other: Complex<T>) -> Complex<T> {
        Complex {
            real: self.real + other.real,
            imaginary: self.imaginary + other.imaginary,
        }
    }
}

impl<T> std::ops::AddAssign for Complex<T>
where
    T: Copy + std::ops::Add<Output = T>
{
    fn add_assign(&mut self, other: Complex<T>) {
        self.real = self.real + other.real;
        self.imaginary = self.imaginary + other.imaginary;
    }
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

    #[test]
    fn test_conj() {
        let c: Complex<f64> = Complex::new(2.0, 4.0);
        let conj: Complex<f64> = c.conj();
        assert_eq!(conj.real, 2.0);
        assert_eq!(conj.imaginary, -4.0);

        let c: Complex<f64> = Complex::new(2.0, -10.0);
        let conj: Complex<f64> = c.conj();
        assert_eq!(conj.real, 2.0);
        assert_eq!(conj.imaginary, 10.0);

        let c: Complex<f64> = Complex::new(2.0, 0.0);
        let conj: Complex<f64> = c.conj();
        assert_eq!(conj.real, 2.0);
        assert_eq!(conj.imaginary, 0.0);
    }

    #[test]
    fn test_re_im() {
        let c: Complex<f64> = Complex::new(42.0, -15.0);
        assert_eq!(c.re(), 42.0);
        assert_eq!(c.im(), -15.0);
    }

    #[test]
    fn test_polar() {
        let c: Complex<f64> = Complex::from_polar(10.0, std::f64::consts::PI / 2.0);
        
        assert!(numbers::approx_equal(c.real, 0.0, 0.0000001));
        assert!(numbers::approx_equal(c.imaginary, 10.0, 0.0000001));
    }

    #[test]
    fn test_neg() {
        let c: Complex<f64> = Complex::new(1.0, 3.0);
        let n: Complex<f64> = -c;
        assert_eq!(n.real, -1.0);
        assert_eq!(n.imaginary, -3.0);
    }

    #[test]
    fn test_add() {
        let a: Complex<f64> = Complex::new(1.0, 2.0);
        let b: Complex<f64> = Complex::new(-3.0, 4.0);
        let c: Complex<f64> = a + b;
        assert_eq!(c.real, -2.0);
        assert_eq!(c.imaginary, 6.0);
    }

    #[test]
    fn test_add_assign() {
        let mut a: Complex<f64> = Complex::new(1.0, 2.0);
        let b: Complex<f64> = Complex::new(-3.0, 4.0);
        a += b;
        assert_eq!(a.real, -2.0);
        assert_eq!(a.imaginary, 6.0);
    }
}
