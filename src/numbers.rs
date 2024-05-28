pub fn approx_equal(x: f64, y: f64, epsilon: f64) -> bool {
    return (x - y).abs() < epsilon;
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
    fn test_sieve_of_eratosthenes() {
        let n: usize = 30;
        let primes: Vec<usize> = sieve_of_eratosthenes(n);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    }
}
