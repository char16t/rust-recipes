pub fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let mut result: usize = 1;
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }

    result
}

pub fn factorial(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    
    let mut result: usize = 1;
    for i in 2..=n {
        result *= i;
    }
    
    result
}

pub fn multinomial_coefficient(n: usize, k: &[usize]) -> usize {
    let mut denominator: usize = 1;
    for &x in k {
        denominator *= factorial(x);
    }
    
    factorial(n) / denominator
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_binomial_coefficient() {
        let n: usize = 5;
        let k: usize = 3;
        let r: usize = binomial_coefficient(n, k);
        assert_eq!(r, 10);

        let n: usize = 3;
        let k: usize = 3;
        let r: usize = binomial_coefficient(n, k);
        assert_eq!(r, 1);

        let n: usize = 3;
        let k: usize = 5;
        let r: usize = binomial_coefficient(n, k);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_multinomial_coefficient() {
        let r: usize = multinomial_coefficient(3, &vec![2, 0, 1]);
        assert_eq!(r, 3);

        let r: usize = multinomial_coefficient(3, &vec![1, 1, 1]);
        assert_eq!(r, 6);
    }
}
