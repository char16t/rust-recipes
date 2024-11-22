pub fn approx_equal(x: f64, y: f64, epsilon: f64) -> bool {
    (x - y).abs() < epsilon
}

pub fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut x: usize = 2;
    while x * x <= n {
        if n % x == 0 {
            return false;
        }
        x += 1
    }
    true
}

pub fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut f: Vec<usize> = Vec::new();
    let mut x: usize = 2;
    while x * x <= n {
        while n % x == 0 {
            f.push(x);
            n /= x;
        }
        x += 1;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

pub fn gcd(a: usize, b: usize) -> usize {
    let mut aa: usize = a;
    let mut bb: usize = b;
    while bb != 0 {
        let temp: usize = bb;
        bb = aa % bb;
        aa = temp;
    }
    aa
}

pub fn gcd_extended(a: i32, b: i32) -> (i32, i32, i32) {
    let mut aa: i32 = a;
    let mut bb: i32 = b;

    let mut x: i32 = 1;
    let mut y: i32 = 0;
    let mut g: i32 = aa;

    while bb != 0 {
        let temp_a: i32 = aa;
        let temp_b: i32 = bb;
        aa = bb;
        bb = temp_a % bb;

        let temp_x: i32 = x;
        let temp_y: i32 = y;
        x = y;
        y = temp_x - (temp_a / temp_b) * y;
        g = temp_a * temp_x + temp_b * temp_y;
    }

    (x, y, g)
}

pub fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

pub fn sieve_of_eratosthenes(n: usize) -> Vec<usize> {
    let mut primes: Vec<bool> = vec![true; n + 1];
    let mut p: usize = 2;

    while p * p <= n {
        if primes[p] {
            let mut i = p * p;
            while i <= n {
                primes[i] = false;
                i += p;
            }
        }
        p += 1;
    }

    let mut result: Vec<usize> = Vec::new();
    for (num, &is_prime) in primes.iter().enumerate().skip(2) {
        if is_prime {
            result.push(num);
        }
    }

    result
}

/// x^n mod m
pub fn modpow(x: usize, n: usize, m: usize) -> usize {
    let mut result: usize = 1;
    let mut base: usize = x;
    let mut exp: usize = n;
    let modulo: usize = m;

    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % modulo;
        }
        base = base * base % modulo;
        exp /= 2;
    }

    result
}

/// x^n
pub fn pow<T1, T2>(x: T1, n: T2) -> T1
where
    T1: Copy
        + PartialEq
        + std::ops::Mul<Output = T1>
        + std::ops::Div<Output = T1>
        + std::ops::Rem<Output = T1>
        + From<u8>,
    T2: Copy
        + PartialEq
        + std::ops::Mul<Output = T2>
        + std::ops::Div<Output = T2>
        + std::ops::Rem<Output = T2>
        + From<u8>,
{
    let mut result: T1 = T1::from(1);
    let mut base: T1 = x;
    let mut exp: T2 = n;

    while exp != T2::from(0) {
        if exp % T2::from(2) == T2::from(1) {
            result = result * base;
        }
        base = base * base;
        exp = exp / T2::from(2);
    }

    result
}

/// Euler's totient function
fn euler_totient(n: u64) -> u64 {
    let mut nn = n;
    let mut result = n;
    let mut i = 2;

    while i * i <= nn {
        if nn % i == 0 {
            while nn % i == 0 {
                nn /= i;
            }
            result -= result / i;
        }
        i += 1;
    }

    if nn > 1 {
        result -= result / nn;
    }

    result
}

pub fn modinverse(x: usize, m: usize) -> Option<usize> {
    if m == 0 {
        return None;
    }
    if gcd(x, m) != 1 {
        return None;
    }

    let phi_m: u64 = euler_totient(m as u64);

    let x_inv: usize = modpow(x, phi_m as usize - 1, m);

    Some(x_inv)
}

pub fn diophantine_equation(a: i32, b: i32, c: i32) -> Option<(i32, i32)> {
    let (x0, y0, gcd) = gcd_extended(a, b);

    if c % gcd != 0 {
        return None;
    }

    let k = c / gcd;
    let x = x0 * k;
    let y = y0 * k;

    Some((x, y))
}

pub fn chinese_remainder_theorem(residues: &[usize], modulii: &[usize]) -> Option<usize> {
    let product: usize = modulii.iter().product::<usize>();
    let mut sum: usize = 0;

    for (&residue, &modulus) in residues.iter().zip(modulii) {
        let p: usize = product / modulus;

        if let Some(inv) = modinverse(p, modulus) {
            sum += residue * inv * p;
        } else {
            return None;
        }
    }

    Some(sum % product)
}

pub fn fib(n: u32) -> u64 {
    let sqrt5: f64 = 2.23606797749979_f64;
    let phi: f64 = (1.0 + sqrt5) / 2.0;
    let psi: f64 = (1.0 - sqrt5) / 2.0;

    ((phi.powf(n as f64) - psi.powf(n as f64)) / sqrt5).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_equal() {
        assert!(approx_equal(0.001, 0.002, 0.01));
        assert!(approx_equal(-0.001, 0.002, 0.01));
        assert!(approx_equal(-0.001, -0.002, 0.01));
        assert!(approx_equal(0.000024115, 0.000023115, 0.00001));
        assert!(!approx_equal(0.000024115, 0.000013115, 0.00001));
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(is_prime(17));
        assert!(is_prime(19));
        assert!(is_prime(23));
        assert!(is_prime(29));

        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(18));
    }

    #[test]
    fn test_prime_factors() {
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(3), vec![3]);
        assert_eq!(prime_factors(4), vec![2, 2]);
        assert_eq!(prime_factors(5), vec![5]);
        assert_eq!(prime_factors(6), vec![2, 3]);
        assert_eq!(prime_factors(32), vec![2, 2, 2, 2, 2]);
        assert_eq!(prime_factors(34), vec![2, 17]);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(18, 12), 6);

        assert_eq!(gcd(17, 23), 1);
        assert_eq!(gcd(24, 36), 12);
        assert_eq!(gcd(1071, 462), 21);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn test_gcd_extended() {
        assert_eq!(gcd_extended(30, 12), (1, -2, 6));
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(30, 12), 60);
        assert_eq!(lcm(12, 30), 60);
    }

    #[test]
    fn test_sieve_of_eratosthenes() {
        let n: usize = 30;
        let primes: Vec<usize> = sieve_of_eratosthenes(n);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    }

    #[test]
    fn test_modpow() {
        let result: usize = modpow(23895, 15, 14189); // 23895^15 % 14189
        assert_eq!(result, 344);
    }

    #[test]
    fn test_pow() {
        assert_eq!(pow(2, 0), 1);
        assert_eq!(pow(2, 4), 16);
        assert_eq!(pow(2, 5), 32);
        assert_eq!(pow(1, 17), 1);
    }

    #[test]
    fn test_euler_totient() {
        assert_eq!(euler_totient(10), 4);
        assert_eq!(euler_totient(6), 2);
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(0), 0);
    }

    #[test]
    fn test_modinverse() {
        assert_eq!(modinverse(3, 7), Some(5));
        assert_eq!((3 * 5) % 7, 1);

        assert_eq!(modinverse(6, 17), Some(3));
        assert_eq!((6 * 3) % 17, 1);

        assert_eq!(modinverse(2, 6), None);
        assert_eq!(modinverse(2, 0), None);
    }

    #[test]
    fn test_diophantine_equation() {
        // 5x + 2y = 11
        let solution: Option<(i32, i32)> = diophantine_equation(5, 2, 11);
        assert_eq!(solution, Some((11, -22)));
        assert_eq!(11, 5 * 11 + 2 * (-22));

        let mut x = 11;
        let mut y = -22;
        let gcd = gcd_extended(5, 2).2;

        for i in 0..10 {
            x += i * 2 / gcd;
            y -= i * 5 / gcd;
            assert_eq!(11, 5 * x + 2 * y);
        }

        for i in 0..10 {
            x -= i * 2 / gcd;
            y += i * 5 / gcd;
            assert_eq!(11, 5 * x + 2 * y);
        }

        // a^n + b^n = c^n, a,b,c=3,4,5, n > 2
        // 3^3*x + 4^3*y = 5^3
        let solution: Option<(i32, i32)> = diophantine_equation(27, 64, 125);
        assert_eq!(solution, None);
    }

    #[test]
    fn test_chinese_remainder_theorem() {
        let residues: Vec<usize> = vec![3, 4, 2];
        let modulii: Vec<usize> = vec![5, 7, 3];
        let result: Option<usize> = chinese_remainder_theorem(&residues, &modulii);
        assert_eq!(result, Some(53));

        // other solutions
        // 53 + 5 * 7 * 3
        // 53 + 5 * 7 * 3 + 5 * 7 * 3
        // ...

        // None because gcd(9, 3) != 1
        let residues: Vec<usize> = vec![3, 4, 2];
        let modulii: Vec<usize> = vec![5, 9, 3];
        let result: Option<usize> = chinese_remainder_theorem(&residues, &modulii);
        assert_eq!(result, None);
    }

    #[test]
    fn test_fib() {
        assert_eq!(fib(0), 0);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(2), 1);
        assert_eq!(fib(3), 2);
        assert_eq!(fib(9), 34);
        assert_eq!(fib(10), 55);
        assert_eq!(fib(14), 377);
    }
}
