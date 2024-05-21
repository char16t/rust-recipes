pub fn binary_search<T>(array: &[T], x: T) -> (bool, usize) where T: PartialOrd {
    let n: usize = array.len();
    let mut k: usize = 0;
    let mut b: usize = n / 2;
    while b >= 1 {
        while k + b < n && array[k + b] <= x {
            k += b;
        }
        b /= 2;
    }
    if array.get(k) == Some(&x) { (true, k) } else { (false, 0) }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_search_1() {
        let a: Vec<i32> = vec![3, 4, 5, 6, 10, 12, 48, 56];
        let (success, position): (bool, usize) = binary_search(&a, 4);
        assert_eq!(success, true);
        assert_eq!(position, 1);
    }

    #[test]
    fn test_binary_search_2() {
        let a: Vec<i32> = vec![3, 4, 5, 6, 10, 12, 48, 56];
        let (success, position): (bool, usize) = binary_search(&a, 8);
        assert_eq!(success, false);
        assert_eq!(position, 0);
    }
}
