use crate::{
    combinatorics,
    numbers::{approx_equal, pow},
};

/// Approximates the sine of an angle using the Taylor series expansion
pub fn sin<T: Into<f64> + Copy>(x: T) -> f64 {
    let x = x.into();
    let mut result: f64 = 0.0;
    for n in 0..16 {
        let sign: f64 = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term =
            sign * x.powi(2 * n + 1) / combinatorics::factorial_i128((2 * n + 1).into()) as f64;
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

#[derive(Clone, Copy)]
pub struct Complex<T> {
    real: T,
    imaginary: T,
}
impl<T> Complex<T>
where
    f64: From<T>,
    T: Copy + std::convert::From<f64> + std::ops::Mul<Output = T> + std::ops::Neg<Output = T>,
{
    pub fn new(real: T, imaginary: T) -> Self {
        Self { real, imaginary }
    }
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            real: r * cos(theta).into(),
            imaginary: r * sin(theta).into(),
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
        Self {
            real: self.real,
            imaginary: -self.imaginary,
        }
    }
}
impl<T> std::ops::Neg for Complex<T>
where
    T: std::ops::Neg<Output = T>,
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
    T: std::ops::Add<Output = T>,
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
    T: Copy + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, other: Complex<T>) {
        self.real = self.real + other.real;
        self.imaginary = self.imaginary + other.imaginary;
    }
}

impl<T> std::ops::Sub for Complex<T>
where
    T: std::ops::Sub<Output = T>,
{
    type Output = Complex<T>;

    fn sub(self, other: Complex<T>) -> Complex<T> {
        Complex {
            real: self.real - other.real,
            imaginary: self.imaginary - other.imaginary,
        }
    }
}

impl<T> std::ops::SubAssign for Complex<T>
where
    T: Copy + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, other: Complex<T>) {
        self.real = self.real - other.real;
        self.imaginary = self.imaginary - other.imaginary;
    }
}

impl<T> std::ops::Mul for Complex<T>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
{
    type Output = Complex<T>;

    fn mul(self, other: Complex<T>) -> Complex<T> {
        let new_real: T = self.real * other.real - self.imaginary * other.imaginary;
        let new_imaginary: T = self.real * other.imaginary + self.imaginary * other.real;

        Complex {
            real: new_real,
            imaginary: new_imaginary,
        }
    }
}

impl<T> std::ops::MulAssign for Complex<T>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, other: Self) {
        let new_real: T = self.real * other.real - self.imaginary * other.imaginary;
        let new_imaginary: T = self.real * other.imaginary + self.imaginary * other.real;

        self.real = new_real;
        self.imaginary = new_imaginary;
    }
}

impl<T> std::ops::Div for Complex<T>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
{
    type Output = Complex<T>;

    fn div(self, other: Complex<T>) -> Complex<T> {
        let denominator: T = other.real * other.real + other.imaginary * other.imaginary;
        let new_real: T = (self.real * other.real + self.imaginary * other.imaginary) / denominator;
        let new_imaginary: T =
            (self.imaginary * other.real - self.real * other.imaginary) / denominator;

        Complex {
            real: new_real,
            imaginary: new_imaginary,
        }
    }
}

impl<T> std::ops::DivAssign for Complex<T>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
{
    fn div_assign(&mut self, other: Complex<T>) {
        let denominator: T = other.real * other.real + other.imaginary * other.imaginary;
        let new_real: T = (self.real * other.real + self.imaginary * other.imaginary) / denominator;
        let new_imaginary: T =
            (self.imaginary * other.real - self.real * other.imaginary) / denominator;

        self.real = new_real;
        self.imaginary = new_imaginary;
    }
}

impl<T> std::fmt::Display for Complex<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {}i", self.real, self.imaginary)
    }
}

impl<T> std::fmt::Debug for Complex<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Complex {{ real: {:?}, imaginary: {:?} }}",
            self.real, self.imaginary
        )
    }
}

pub fn cross_product(a: (f64, f64), b: (f64, f64)) -> f64 {
    let c1: Complex<f64> = Complex::new(a.0, -a.1);
    let c2: Complex<f64> = Complex::new(b.0, b.1);
    (c1 * c2).imaginary
}

/// Point_position_relative_line
/// The line passes through the points s1 and s2. We look from point s1 in the direction of s2.
/// Returns 1 if point P left, and -1 if point P right
pub fn point_position_relative_line(s1: (f64, f64), s2: (f64, f64), p: (f64, f64)) -> f64 {
    let cs1: Complex<f64> = Complex::new(s1.0, s1.0);
    let cs2: Complex<f64> = Complex::new(s2.0, s2.1);
    let cp: Complex<f64> = Complex::new(p.0, p.1);

    let a: Complex<f64> = cp - cs1;
    let b: Complex<f64> = cp - cs2;

    let r: f64 = cross_product((a.real, a.imaginary), (b.real, b.imaginary));
    if approx_equal(r, 0.0, 0.00001) {
        0.0
    } else {
        r.signum()
    }
}

pub fn distance_from_point_to_line(s1: (f64, f64), s2: (f64, f64), p: (f64, f64)) -> f64 {
    let cs1: Complex<f64> = Complex::new(s1.0, s1.1);
    let cs2: Complex<f64> = Complex::new(s2.0, s2.1);
    let cp: Complex<f64> = Complex::new(p.0, p.1);

    let a: Complex<f64> = cs1 - cp;
    let b: Complex<f64> = cs2 - cp;
    let product: f64 = cross_product((a.real, a.imaginary), (b.real, b.imaginary));

    let dist_s1_s2: f64 = f64::sqrt(pow(s2.0 - s1.0, 2) + pow(s2.1 - s1.1, 2));

    (product / dist_s1_s2).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers;

    #[test]
    fn test_sin() {
        assert!(numbers::approx_equal(
            sin(std::f64::consts::PI),
            0.0,
            0.0000001
        ));
        assert!(numbers::approx_equal(
            sin(std::f64::consts::PI / 2.0),
            1.0,
            0.0000001
        ));
        assert!(numbers::approx_equal(
            sin(std::f64::consts::PI / 4.0),
            std::f64::consts::FRAC_1_SQRT_2, // 0.70710678119
            0.0000001
        ));
        assert!(numbers::approx_equal(
            sin(std::f64::consts::PI * 2.0),
            0.0,
            0.0000001
        ));
    }

    #[test]
    fn test_cos() {
        assert!(numbers::approx_equal(
            cos(std::f64::consts::PI),
            -1.0,
            0.0000001
        ));
        assert!(numbers::approx_equal(
            cos(std::f64::consts::PI / 2.0),
            0.0,
            0.0000001
        ));
        assert!(numbers::approx_equal(
            cos(std::f64::consts::PI / 4.0),
            std::f64::consts::FRAC_1_SQRT_2, // 0.70710678119
            0.0000001
        ));
        assert!(numbers::approx_equal(
            cos(std::f64::consts::PI * 2.0),
            1.0,
            0.0000001
        ));
    }

    #[test]
    fn test_exp() {
        assert!(numbers::approx_equal(
            exp::<f64>(1.0),
            std::f64::consts::E, // 2.71828182846
            0.000000000001
        ));
        assert!(numbers::approx_equal(
            exp::<f64>(2.0),
            7.38905609893065,
            0.000000000001
        ));
        assert!(numbers::approx_equal(
            exp::<f64>(2.5),
            12.182493960703473,
            0.000000000001
        ));
        assert!(numbers::approx_equal(
            exp::<f64>(5.0),
            148.4131591025766,
            0.000000000001
        ));
        assert!(numbers::approx_equal(
            exp::<f64>(-5.0),
            0.006737946999085467,
            0.000000000001
        ));
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

    #[test]
    fn test_sub() {
        let a: Complex<f64> = Complex::new(1.0, 2.0);
        let b: Complex<f64> = Complex::new(-3.0, 4.0);
        let c: Complex<f64> = a - b;
        assert_eq!(c.real, 4.0);
        assert_eq!(c.imaginary, -2.0);
    }

    #[test]
    fn test_sub_assign() {
        let mut a: Complex<f64> = Complex::new(1.0, 2.0);
        let b: Complex<f64> = Complex::new(-3.0, 4.0);
        a -= b;
        assert_eq!(a.real, 4.0);
        assert_eq!(a.imaginary, -2.0);
    }

    #[test]
    fn test_mul() {
        let a: Complex<f64> = Complex::new(3.0, 4.0);
        let b: Complex<f64> = Complex::new(1.0, 2.0);
        let c: Complex<f64> = a * b;
        assert_eq!(c.real, -5.0);
        assert_eq!(c.imaginary, 10.0);
    }

    #[test]
    fn test_mul_assign() {
        let mut a: Complex<f64> = Complex::new(3.0, 4.0);
        let b: Complex<f64> = Complex::new(1.0, 2.0);
        a *= b;
        assert_eq!(a.real, -5.0);
        assert_eq!(a.imaginary, 10.0);
    }

    #[test]
    fn test_div() {
        let a: Complex<f64> = Complex::new(3.0, 4.0);
        let b: Complex<f64> = Complex::new(1.0, 2.0);
        let c: Complex<f64> = a / b;
        assert_eq!(c.real, 2.2);
        assert_eq!(c.imaginary, -0.4);
    }

    #[test]
    fn test_div_assign() {
        let mut a: Complex<f64> = Complex::new(3.0, 4.0);
        let b: Complex<f64> = Complex::new(1.0, 2.0);
        a /= b;
        assert_eq!(a.real, 2.2);
        assert_eq!(a.imaginary, -0.4);
    }

    #[test]
    fn test_display() {
        let complex_number: Complex<f64> = Complex::new(3.15, 4.0);
        assert_eq!(format!("{}", complex_number), "3.15 + 4i");
    }

    #[test]
    fn test_debug() {
        let complex_number: Complex<f64> = Complex::new(1.5f64, 2.5f64);
        assert_eq!(
            format!("{:?}", complex_number),
            "Complex { real: 1.5, imaginary: 2.5 }"
        );
    }

    #[test]
    fn test_cross_product() {
        let actual: f64 = cross_product((4.0, 2.0), (1.0, 2.0));
        assert_eq!(actual, 6.0);
    }

    #[test]
    fn test_point_position_relative_line() {
        let actual: f64 = point_position_relative_line((2.0, 2.0), (5.0, 5.0), (1.0, 3.0));
        assert_eq!(actual, 1.0);

        let actual: f64 = point_position_relative_line((5.0, 5.0), (2.0, 2.0), (1.0, 3.0));
        assert_eq!(actual, -1.0);

        let actual: f64 = point_position_relative_line((5.0, 5.0), (2.0, 2.0), (3.0, 3.0));
        assert_eq!(actual, 0.0);
    }

    #[test]
    fn test_distance_from_point_to_line() {
        let distance: f64 = distance_from_point_to_line((2.0, -1.0), (-2.0, 1.0), (2.0, 2.0));
        assert!(approx_equal(distance, 2.6832815729997477, 0.000001));

        let distance: f64 = distance_from_point_to_line((2.0, -1.0), (-2.0, 1.0), (10.0, 20.0));
        assert!(approx_equal(distance, 22.360679774997898, 0.000001));
    }
}
