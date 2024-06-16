pub fn approx_equal(x: f64, y: f64, epsilon: f64) -> bool {
    return (x - y).abs() < epsilon;
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
    let mut base: usize = x as usize;
    let mut exp: usize = n as usize;
    let modulo: usize = m as usize;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base % modulo) as usize;
        }
        base = (base * base % modulo) as usize;
        exp /= 2;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_equal() {
        assert!(approx_equal(0.001, 0.002, 0.01) == true);
        assert!(approx_equal(-0.001, 0.002, 0.01) == true);
        assert!(approx_equal(-0.001, -0.002, 0.01) == true);
        assert!(approx_equal(0.000024115, 0.000023115, 0.00001) == true);
        assert!(approx_equal(0.000024115, 0.000013115, 0.00001) == false);
    }

    #[test]
    fn test_is_prime() {
        assert_eq!(is_prime(2), true);
        assert_eq!(is_prime(3), true);
        assert_eq!(is_prime(5), true);
        assert_eq!(is_prime(7), true);
        assert_eq!(is_prime(11), true);
        assert_eq!(is_prime(13), true);
        assert_eq!(is_prime(17), true);
        assert_eq!(is_prime(19), true);
        assert_eq!(is_prime(23), true);
        assert_eq!(is_prime(29), true);

        assert_eq!(is_prime(0), false);
        assert_eq!(is_prime(1), false);
        assert_eq!(is_prime(4), false);
        assert_eq!(is_prime(18), false);
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
        assert_eq!(gcd_extended(30, 12), (1, -2,  6));
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
    fn test_euler_totient() {
        assert_eq!(euler_totient(10), 4);
        assert_eq!(euler_totient(6), 2);
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(0), 0);
    }
}
