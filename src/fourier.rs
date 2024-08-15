use std::f64::consts::PI;
use crate::geometry::Complex;

pub fn fast_fourier_transform(a: Vec<Complex<f64>>, d: i32) -> Vec<Complex<f64>> {
    let n: usize = a.len();
    let mut r: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n {
        let mut b: usize = 0;

        let mut z: usize = 1;
        while z < n {
            b *= 2;
            if k & z != 0 {
                b += 1;
            }
            z *= 2;
        }
        r[b] = a[k];
    }

    let mut m: usize = 2;
    while m <= n {
        let wm: Complex<f64> = Complex::new(0.0, d as f64 * 2.0 * PI / m as f64).exp();
        for k in (0..n).step_by(m) {
            let mut w: Complex<f64> = Complex::new(1.0, 0.0);
            for j in 0..m / 2 {
                let u: Complex<f64> = r[k + j];
                let t: Complex<f64> = w * r[k + j + m / 2];
                r[k + j] = u + t;
                r[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
        m *= 2;
    }

    if d == -1 {
        for i in 0..n {
            r[i] /= Complex::new(n as f64, 0.0);
        }
    }

    r
}

pub fn multiply_polynomials(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut aa: Vec<Complex<f64>> = a.iter().map(|&n| { Complex::new(n as f64, 0.0) }).collect();
    let mut bb: Vec<Complex<f64>> = b.iter().map(|&n| { Complex::new(n as f64, 0.0) }).collect();
    let n: usize =  2usize.pow(((aa.len() + bb.len() - 1) as f64).log2().ceil() as u32);

    let zeros_to_append_a: usize = n - aa.len();
    aa.extend(std::iter::repeat(Complex::new(0.0, 0.0)).take(zeros_to_append_a));

    let zeros_to_append_b: usize = n - bb.len();
    bb.extend(std::iter::repeat(Complex::new(0.0, 0.0)).take(zeros_to_append_b));

    let ta: Vec<Complex<f64>> = fast_fourier_transform(aa, 1);
    let tb: Vec<Complex<f64>> = fast_fourier_transform(bb, 1);

    let mut tr: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        tr[i] = ta[i] * tb[i];
    }
    let p: Vec<Complex<f64>> = fast_fourier_transform(tr,-1);
    let r: Vec<i64> = p.iter().take(a.len() + b.len() - 1).map(|&c| { (c.re().signum() * (c.re().abs() + 0.5)) as i64}).collect();
    r
}

#[cfg(test)]
mod tests {
    use crate::numbers;
    use super::*;

    #[test]
    fn test_fast_fourier_transform() {
        let f: Vec<Complex<f64>> = vec![Complex::new(3.0, 0.0), Complex::new(2.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];        
        let tf: Vec<Complex<f64>> = fast_fourier_transform(f, 1);

        assert!(numbers::approx_equal(tf[0].re(), 5.0, 0.0000000001));
        assert!(numbers::approx_equal(tf[0].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(tf[1].re(), 3.0, 0.0000000001));
        assert!(numbers::approx_equal(tf[1].im(), 2.0, 0.0000000001));

        assert!(numbers::approx_equal(tf[2].re(), 1.0, 0.0000000001));
        assert!(numbers::approx_equal(tf[2].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(tf[3].re(), 3.0, 0.0000000001));
        assert!(numbers::approx_equal(tf[3].im(), -2.0, 0.0000000001));

        
        let g: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(5.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let tg: Vec<Complex<f64>> = fast_fourier_transform(g, 1);

        assert!(numbers::approx_equal(tg[0].re(), 6.0, 0.0000000001));
        assert!(numbers::approx_equal(tg[0].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(tg[1].re(), 1.0, 0.0000000001));
        assert!(numbers::approx_equal(tg[1].im(), 5.0, 0.0000000001));

        assert!(numbers::approx_equal(tg[2].re(), -4.0, 0.0000000001));
        assert!(numbers::approx_equal(tg[2].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(tg[3].re(), 1.0, 0.0000000001));
        assert!(numbers::approx_equal(tg[3].im(), -5.0, 0.0000000001));

        let n: usize = 4;
        let mut tp: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            tp[i] = tf[i] * tg[i];
        }
        let p: Vec<Complex<f64>> = fast_fourier_transform(tp,-1);

        assert!(numbers::approx_equal(p[0].re(), 3.0, 0.0000000001));
        assert!(numbers::approx_equal(p[0].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(p[1].re(), 17.0, 0.0000000001));
        assert!(numbers::approx_equal(p[1].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(p[2].re(), 10.0, 0.0000000001));
        assert!(numbers::approx_equal(p[2].im(), 0.0, 0.0000000001));

        assert!(numbers::approx_equal(p[3].re(), 0.0, 0.0000000001));
        assert!(numbers::approx_equal(p[3].im(), 0.0, 0.0000000001));

        let r: Vec<i64> = p.iter().map(|&c| { (c.re() + 0.5) as i64}).collect();
        assert_eq!(r, vec![3, 17, 10, 0])
    }

    #[test]
    fn test_multiply_polynomials() {
        // f(x) = 2x + 3
        // g(x) = 5x + 1
        let r: Vec<i64> = multiply_polynomials(&[3, 2], &[1, 5]);
        // f(x) * g(x) = 10x^2 + 17x + 3
        assert_eq!(r, vec![3, 17, 10]);


        // f(x) = 4x^2 + 3x - 1
        // g(x) = 5x - 1
        let r: Vec<i64> = multiply_polynomials(&[-1, 3, 4], &[-1, 5]);
        // f(x) * g(x) = 20x^3 + 11x^2 - 8x + 1 
        assert_eq!(r, vec![1, -8, 11, 20]);
    }
}
