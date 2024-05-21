use crate::coordinates;

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

pub fn build_prefix_sum_array_2d<T>(arr: &[T], rows: usize, cols: usize) -> Vec<T>
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy 
{
    let mut prefix_sum: Vec<T> = vec![T::default(); arr.len()];
    let xy = coordinates::create_coordinate_function_2d!(rows, cols);
    
    // Fill first row
    prefix_sum[xy(0, 0)] = arr[xy(0, 0)];
    for i in 1..cols {
        prefix_sum[xy(0, i)] = prefix_sum[xy(0, i - 1)] + arr[xy(0, i)];
    }

    // Fill first column
    for i in 1..rows {
        prefix_sum[xy(i, 0)] = prefix_sum[xy(i - 1, 0)] + arr[xy(i ,0)];
    }

    for i in 1..rows {
        for j in 1..cols {
            prefix_sum[xy(i, j)] = 
                prefix_sum[xy(i-1, j)]
                + prefix_sum[xy(i, j-1)] 
                - prefix_sum[xy(i-1, j-1)] 
                + arr[xy(i, j)]
        }
    }

    prefix_sum
}

pub fn range_sum_2d<T>(arr: &[T], rows: usize, cols: usize, top_left: (usize, usize), bottom_right: (usize, usize)) -> T 
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy
{
    // * +-----------+
    // * |  D     C  |
    // * |   +---+   |
    // * |   |   |   |
    // * | B +---A   |
    // * +-----------+
    let xy = coordinates::create_coordinate_function_2d!(rows, cols);
    if top_left.0 == 0 && top_left.1 == 0 {
        let a: usize = xy(bottom_right.0, bottom_right.1);
        return arr[a];
    } else if top_left.0 == 0 {
        let a: usize = xy(bottom_right.0, bottom_right.1);
        let b: usize = xy(bottom_right.0, top_left.1 - 1);
        return arr[a] - arr[b];
    } else if top_left.1 == 0 {
        let a: usize = xy(bottom_right.0, bottom_right.1);
        let c: usize = xy(top_left.0 - 1,  bottom_right.1);
        return arr[a] - arr[c];
    } else {
        let a: usize = xy(bottom_right.0, bottom_right.1);
        let b: usize = xy(bottom_right.0, top_left.1 - 1);
        let c: usize = xy(top_left.0 - 1,  bottom_right.1);
        let d: usize = xy(top_left.0 - 1, top_left.1 - 1);
        return arr[a] - arr[b] - arr[c] + arr[d]
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

    #[test]
    fn test_build_prefix_sum_array_2d() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_2d(&original_array, 4, 5);
        assert_eq!(prefix_sum_array.len(), 20);
        
        // 1, 2,  3,  4,  5, 
        // 2, 4,  6,  8, 10, 
        // 3, 6,  9, 12, 15, 
        // 4, 8, 12, 16, 20
        assert_eq!(prefix_sum_array[0], 1);
        assert_eq!(prefix_sum_array[1], 2);
        assert_eq!(prefix_sum_array[2], 3);
        assert_eq!(prefix_sum_array[3], 4);
        assert_eq!(prefix_sum_array[4], 5);

        assert_eq!(prefix_sum_array[5], 2);
        assert_eq!(prefix_sum_array[6], 4);
        assert_eq!(prefix_sum_array[7], 6);
        assert_eq!(prefix_sum_array[8], 8);
        assert_eq!(prefix_sum_array[9], 10);

        assert_eq!(prefix_sum_array[10], 3);
        assert_eq!(prefix_sum_array[11], 6);
        assert_eq!(prefix_sum_array[12], 9);
        assert_eq!(prefix_sum_array[13], 12);
        assert_eq!(prefix_sum_array[14], 15);

        assert_eq!(prefix_sum_array[15], 4);
        assert_eq!(prefix_sum_array[16], 8);
        assert_eq!(prefix_sum_array[17], 12);
        assert_eq!(prefix_sum_array[18], 16);
        assert_eq!(prefix_sum_array[19], 20);
    }

    #[test]
    fn test_range_sum_2d() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_2d(&original_array, 4, 5);

        let sum: i32 = range_sum_2d(&prefix_sum_array, 4, 5, (1, 1), (2, 3));
        assert_eq!(sum, 6);

        let sum: i32 = range_sum_2d(&prefix_sum_array, 4, 5, (0, 0), (2, 3));
        assert_eq!(sum, 12);

        let sum: i32 = range_sum_2d(&prefix_sum_array, 4, 5, (1, 0), (2, 3));
        assert_eq!(sum, 8);

        let sum: i32 = range_sum_2d(&prefix_sum_array, 4, 5, (0, 1), (2, 3));
        assert_eq!(sum, 9);
    }
}
