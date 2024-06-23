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
}
