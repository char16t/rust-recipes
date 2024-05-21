pub fn build_prefix_sum_array<T>(arr: &[T]) -> Vec<T> where T: Default + std::ops::Add<Output = T> + Copy {
    let mut prefix_sum: Vec<T> = vec![T::default(); arr.len()];
    prefix_sum[0] = arr[0];
    for i in 1..prefix_sum.len() {
        prefix_sum[i] = prefix_sum[i - 1] + arr[i];
    }
    prefix_sum
}

pub fn range_sum<T>(prefix_sum: &[T], l: usize, r: usize) -> T where T: Default + std::ops::Sub<Output = T> + Copy {
    if l == 0 {
        prefix_sum[r]
    } else {
        prefix_sum[r] - prefix_sum[l-1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prefix_sum_array() {
        let original_array: Vec<i32> = vec![1, 2, 3, 4, 5];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array(&original_array);
        assert_eq!(prefix_sum_array.len(), 5);
        assert_eq!(prefix_sum_array[0], 1);
        assert_eq!(prefix_sum_array[1], 3);
        assert_eq!(prefix_sum_array[2], 6);
        assert_eq!(prefix_sum_array[3], 10);
        assert_eq!(prefix_sum_array[4], 15);
    }

    #[test]
    fn test_range_sum() {
        let original_array: Vec<i32> = vec![1, 2, 3, 4, 5];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array(&original_array);
        
        assert_eq!(range_sum(&prefix_sum_array, 1, 3), 9);
        assert_eq!(range_sum(&prefix_sum_array, 0, 0), 1);
        assert_eq!(range_sum(&prefix_sum_array, 1, 1), 2);
        assert_eq!(range_sum(&prefix_sum_array, 1, 2), 5);
        assert_eq!(range_sum(&prefix_sum_array, 0, 4), 15);
    }
}
