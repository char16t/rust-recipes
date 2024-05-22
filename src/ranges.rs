use std::cmp;
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

pub fn range_sum_2d<T>(arr: &[T], _rows: usize, cols: usize, top_left: (usize, usize), bottom_right: (usize, usize)) -> T 
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy
{
    if top_left.0 > bottom_right.0 {
        panic!(
            "Coordinate X of top left point ({}, {}) should be less or equals of bottom right's coordinate X ({}, {})",
            top_left.0, top_left.1,
            bottom_right.0, bottom_right.1 
        );
    }
    if top_left.1 > bottom_right.1 {
        panic!(
            "Coordinate Y of top left point ({}, {}) should be less or equals of bottom right's coordinate Y ({}, {})",
            top_left.0, top_left.1,
            bottom_right.0, bottom_right.1 
        );
    }

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

pub fn build_prefix_sum_array_3d<T>(arr: &[T], rows: usize, cols: usize, depth: usize) -> Vec<T>
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy 
{
    let mut prefix_sum: Vec<T> = vec![T::default(); arr.len()];
    let xy = coordinates::create_coordinate_function_3d!(rows, cols, depth);
    
    // Filling first row, first col, first depth
    prefix_sum[xy(0, 0, 0)] = arr[xy(0, 0, 0)];
    for i in 1..rows {
        prefix_sum[xy(i, 0, 0)] = prefix_sum[xy(i - 1, 0, 0)] + arr[xy(i, 0, 0)]
    }
    for i in 1..cols {
        prefix_sum[xy(0, i, 0)] = prefix_sum[xy(0, i-1, 0)] + arr[xy(0, i, 0)]
    }
    for i in 1..depth {
        prefix_sum[xy(0, 0, i)] = prefix_sum[xy(0, 0, i-1)] + arr[xy(0, 0, i)]
    }

    // Filling the cells of sides
    for j in 1..cols {
        for k in 1..depth {
            prefix_sum[xy(0, j, k)] = arr[xy(0, j, k)]
                + prefix_sum[xy(0, j-1, k)]
                + prefix_sum[xy(0, j, k-1)]
                - prefix_sum[xy(0, j-1, k-1)]
        }
    }
    for i in 1..rows {
        for k in 1..depth {
            prefix_sum[xy(i, 0, k)] = arr[xy(i, 0, k)]
                + prefix_sum[xy(i-1, 0, k)]
                + prefix_sum[xy(i, 0, k-1)]
                - prefix_sum[xy(i-1, 0, k-1)]
        }
    }
    for i in 1..rows {
        for j in 1..cols {
            prefix_sum[xy(i, j, 0)] = arr[xy(i, j, 0)]
                + prefix_sum[xy(i-1, j, 0)]
                + prefix_sum[xy(i, j-1, 0)]
                - prefix_sum[xy(i-1, j-1, 0)]
        }
    }

    // Fill rest
    for i in 1..rows {
        for j in 1..cols {
            for k in 1..depth {
                prefix_sum[xy(i, j, k)] = arr[xy(i, j, k)]
                    + prefix_sum[xy(i-1, j, k)]
                    + prefix_sum[xy(i, j-1, k)]
                    + prefix_sum[xy(i, j, k-1)]
                    - prefix_sum[xy(i-1, j-1, k)]
                    - prefix_sum[xy(i, j-1, k-1)]
                    - prefix_sum[xy(i-1, j, k-1)]
                    + prefix_sum[xy(i-1, k-1, j-1)] 
            }
        }
    }
    prefix_sum
}

///
/// Point: (x = rows, y = cols, z = depth)
/// 
/// TL - top left
/// BR - bottom right
/// 
//        +---------+
//       /|        /|
//      / |       / |
//  TL •--|------+  |
//     |  +------|--• BR 
//     | /       | /
//     |/        |/
//     +--------+
//
pub fn range_sum_3d<T>(arr: &[T], _rows: usize, cols: usize, depth: usize, top_left: (usize, usize, usize), bottom_right: (usize, usize, usize)) -> T 
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy
{
    if top_left.0 > bottom_right.0 {
        panic!(
            "Coordinate X of top left point ({}, {}, {}) should be less or equals of bottom right's coordinate X ({}, {}, {})",
            top_left.0, top_left.1, top_left.2,
            bottom_right.0, bottom_right.1, bottom_right.2
        );
    }
    if top_left.1 > bottom_right.1 {
        panic!(
            "Coordinate Y of top left point ({}, {}, {}) should be less or equals of bottom right's coordinate Y ({}, {}, {})",
            top_left.0, top_left.1, top_left.2,
            bottom_right.0, bottom_right.1, bottom_right.2
        );
    }
    if top_left.2 > bottom_right.2 {
        panic!(
            "Coordinate Z of top left point ({}, {}, {}) should be less or equals of bottom right's coordinate Z ({}, {}, {})",
            top_left.0, top_left.1, top_left.2,
            bottom_right.0, bottom_right.1, bottom_right.2
        );
    }

    let xy = coordinates::create_coordinate_function_3d!(rows, cols, depth);
    let (x1, y1, z1): (usize, usize, usize) = top_left;
    let (x2, y2, z2): (usize, usize, usize) = bottom_right;
    if x1 == 0 && y1 == 0 && z1 == 0 {
        return arr[xy(x2, y2, z2)];
    } else if x1 == 0 && y1 == 0 {
        return arr[xy(x2, y2, z2)] - arr[xy(x2, y2, z1-1)]
    } else if x1 == 0 && z1 == 0 {
        return arr[xy(x2, y2, z2)] - arr[xy(x2, y1-1, z2)]; 
    } else if y1 == 0 && z1 == 0 {
        return arr[xy(x2, y2, z2)] - arr[xy(x1-1, y2, z2)];
    } else if x1 == 0 {
        return arr[xy(x2, y2, z2)] 
            - arr[xy(x2, y1-1, z2)] 
            - arr[xy(x2, y2, z1-1)]
            + arr[xy(x2, y1-1, z1-1)];
    } else if y1 == 0 {
        return arr[xy(x2, y2, z2)] 
            - arr[xy(x1-1, y2, z2)] 
            - arr[xy(x2, y2, z1-1)]
            + arr[xy(x1-1, y2, z1-1)];
    } else if z1 == 0 {
        return arr[xy(x2, y2, z2)] 
            - arr[xy(x1-1, y2, z2)] 
            - arr[xy(x2, y1-1, z2)] 
            + arr[xy(x1-1, y1-1, z2)];
    } else {
        return arr[xy(x2, y2, z2)] 
            - arr[xy(x1-1, y2, z2)] 
            - arr[xy(x2, y1-1, z2)] 
            - arr[xy(x2, y2, z1-1)]
            + arr[xy(x1-1, y1-1, z2)] 
            + arr[xy(x1-1, y2, z1-1)] 
            + arr[xy(x2, y1-1, z1-1)] 
            - arr[xy(x1-1, y1-1, z1-1)];
    }
}

pub fn build_sparse_table<T>(arr: &[T]) -> Vec<T>
where 
    T: Default + Copy + Ord
{
    let n: usize = arr.len();
    let logn: usize = (n as f64).log2() as usize + 1;
    let mut table: Vec<T> = vec![T::default(); n * logn];
    let xy = coordinates::create_coordinate_function_2d!(logn, n);

    for i in 0..n {
        table[xy(0, i)] = arr[i];
    }

    let mut j: usize = 1;
    while (1 << j) <= n {
        let mut i: usize = 0;
        while i + (1 << j) <= n {
            table[xy(j, i)] = cmp::min(table[xy(j-1, i)], table[xy(j-1, i + (1 << (j-1)))]);
            i += 1;
        }
        j += 1;
    }

    return table;
}

pub fn range_min<T: Copy + Ord>(sparse_table: &[T], length: usize, left: usize, right: usize) -> T {
    let _logn: usize = (length as f64).log2() as usize + 1;
    let k: usize = ((right - left + 1) as f64).log2() as usize;
    let xy = coordinates::create_coordinate_function_2d!(_logn, length);
    cmp::min(sparse_table[xy(k ,left)], sparse_table[xy(k, right + 1 - (1 << k))])
}

pub fn build_fenwick_tree<T>(arr: &[T]) -> Vec<T> 
where T:  Default + Copy + std::ops::AddAssign + std::ops::Add<T, Output = T>
{
    let mut tree: Vec<T> = vec![T::default(); arr.len() + 1];
    for i in 0..arr.len() {
        add_to_fenwick_tree::<T, T>(tree.as_mut_slice(), i+1, arr[i]);
    }
    tree
}

/// increase index-th element to delta. delta can have any sign
pub fn add_to_fenwick_tree<T, D>(fenwick_tree: &mut [T], index: usize, delta: D)
where
    T: std::ops::Add<D, Output = T> + std::ops::AddAssign<D>,
    D: Copy
{
    let mut idx: usize = index;
    while idx < fenwick_tree.len() {
        fenwick_tree[idx] += delta;
        idx = idx + (idx & !idx + 1);
    }
}

pub fn range_sum_fenwick<T>(fenwick_tree: &[T], index: usize) -> T
where
    T: Default + Copy + std::ops::Add<Output = T>
{
    let mut idx: usize = index;
    let mut sum: T = T::default();
    while idx > 0 {
        sum = sum + fenwick_tree[idx];
        idx = idx - (idx & !idx + 1);
    }
    sum
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

    #[test]
    #[should_panic]
    fn test_range_sum_2d_panic_x() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_2d(&original_array, 4, 5);

        range_sum_2d(&prefix_sum_array, 4, 5, (2, 3), (1, 1));
    }

    #[test]
    #[should_panic]
    fn test_range_sum_2d_panic_y() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_2d(&original_array, 4, 5);

        range_sum_2d(&prefix_sum_array, 4, 5, (1, 3), (2, 2));
    }

    #[test]
    fn test_build_prefix_sum_array_3d() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_3d(&original_array, 4, 4, 4);
        /*
            1 2 3 4 
            2 4 6 8 
            3 6 9 12 
            4 8 12 16 

            2 4 6 8 
            4 8 12 16 
            6 12 18 24 
            8 16 24 32

            3 6 9 12 
            6 12 18 24 
            9 18 27 36 
            12 24 36 48 

            4 8 12 16 
            8 16 24 32 
            12 24 36 48 
            16 32 48 64 
        */
        assert_eq!(prefix_sum_array.len(), 64);
        
        assert_eq!(prefix_sum_array[0], 1);
        assert_eq!(prefix_sum_array[1], 2);
        assert_eq!(prefix_sum_array[2], 3);
        assert_eq!(prefix_sum_array[3], 4);
        assert_eq!(prefix_sum_array[4], 2);
        assert_eq!(prefix_sum_array[5], 4);
        assert_eq!(prefix_sum_array[6], 6);
        assert_eq!(prefix_sum_array[7], 8);
        assert_eq!(prefix_sum_array[8], 3);
        assert_eq!(prefix_sum_array[9], 6);
        assert_eq!(prefix_sum_array[10], 9);
        assert_eq!(prefix_sum_array[11], 12);
        assert_eq!(prefix_sum_array[12], 4);
        assert_eq!(prefix_sum_array[13], 8);
        assert_eq!(prefix_sum_array[14], 12);
        assert_eq!(prefix_sum_array[15], 16);

        assert_eq!(prefix_sum_array[16], 2);
        assert_eq!(prefix_sum_array[17], 4);
        assert_eq!(prefix_sum_array[18], 6);
        assert_eq!(prefix_sum_array[19], 8);
        assert_eq!(prefix_sum_array[20], 4);
        assert_eq!(prefix_sum_array[21], 8);
        assert_eq!(prefix_sum_array[22], 12);
        assert_eq!(prefix_sum_array[23], 16);
        assert_eq!(prefix_sum_array[24], 6);
        assert_eq!(prefix_sum_array[25], 12);
        assert_eq!(prefix_sum_array[26], 18);
        assert_eq!(prefix_sum_array[27], 24);
        assert_eq!(prefix_sum_array[28], 8);
        assert_eq!(prefix_sum_array[29], 16);
        assert_eq!(prefix_sum_array[30], 24);
        assert_eq!(prefix_sum_array[31], 32);

        assert_eq!(prefix_sum_array[32], 3);
        assert_eq!(prefix_sum_array[33], 6);
        assert_eq!(prefix_sum_array[34], 9);
        assert_eq!(prefix_sum_array[35], 12);
        assert_eq!(prefix_sum_array[36], 6);
        assert_eq!(prefix_sum_array[37], 12);
        assert_eq!(prefix_sum_array[38], 18);
        assert_eq!(prefix_sum_array[39], 24);
        assert_eq!(prefix_sum_array[40], 9);
        assert_eq!(prefix_sum_array[41], 18);
        assert_eq!(prefix_sum_array[42], 27);
        assert_eq!(prefix_sum_array[43], 36);
        assert_eq!(prefix_sum_array[44], 12);
        assert_eq!(prefix_sum_array[45], 24);
        assert_eq!(prefix_sum_array[46], 36);
        assert_eq!(prefix_sum_array[47], 48);

        assert_eq!(prefix_sum_array[48], 4);
        assert_eq!(prefix_sum_array[49], 8);
        assert_eq!(prefix_sum_array[50], 12);
        assert_eq!(prefix_sum_array[51], 16);
        assert_eq!(prefix_sum_array[52], 8);
        assert_eq!(prefix_sum_array[53], 16);
        assert_eq!(prefix_sum_array[54], 24);
        assert_eq!(prefix_sum_array[55], 32);
        assert_eq!(prefix_sum_array[56], 12);
        assert_eq!(prefix_sum_array[57], 24);
        assert_eq!(prefix_sum_array[58], 36);
        assert_eq!(prefix_sum_array[59], 48);
        assert_eq!(prefix_sum_array[60], 16);
        assert_eq!(prefix_sum_array[61], 32);
        assert_eq!(prefix_sum_array[62], 48);
        assert_eq!(prefix_sum_array[63], 64);
    }

    #[test]
    fn test_range_sum_3d() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_3d(&original_array, 4, 4, 4);
        
        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 0, 0), (0, 0, 0));
        assert_eq!(sum, 1);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 0, 1), (0, 0, 3));
        assert_eq!(sum, 3);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 1, 0), (0, 1, 3));
        assert_eq!(sum, 4);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (1, 0, 0), (1, 1, 3));
        assert_eq!(sum, 8);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (1, 1, 0), (1, 1, 3));
        assert_eq!(sum, 4);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (1, 0, 1), (2, 0, 3));
        assert_eq!(sum, 6);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 1, 1), (2, 1, 3));
        assert_eq!(sum, 9);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (1, 1, 1), (1, 1, 1));
        assert_eq!(sum, 1);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (1, 1, 1), (3, 3, 3));
        assert_eq!(sum, 27);

        let sum: i32 = range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 0, 0), (3, 3, 3));
        assert_eq!(sum, 64);
    }

    #[test]
    #[should_panic]
    fn test_range_sum_3d_panic_x() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_3d(&original_array, 4, 4, 4);
        
        range_sum_3d(&prefix_sum_array, 4, 4, 4, (1, 0, 0), (0, 0, 0));
    }

    #[test]
    #[should_panic]
    fn test_range_sum_3d_panic_y() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_3d(&original_array, 4, 4, 4);
        
        range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 1, 0), (0, 0, 0));
    }

    #[test]
    #[should_panic]
    fn test_range_sum_3d_panic_z() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,

            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ];
        let prefix_sum_array: Vec<i32> = build_prefix_sum_array_3d(&original_array, 4, 4, 4);
        
        range_sum_3d(&prefix_sum_array, 4, 4, 4, (0, 0, 1), (0, 0, 0));
    }

    #[test]
    fn test_build_sparse_table() {
        let arr: Vec<usize> = vec![1, 3, 4, 8, 6, 1, 4, 2];
        let table: Vec<usize> = build_sparse_table(&arr);
        assert_eq!(table.len(), 32);

        assert_eq!(table[0], 1);
        assert_eq!(table[1], 3);
        assert_eq!(table[2], 4);
        assert_eq!(table[3], 8);
        assert_eq!(table[4], 6);
        assert_eq!(table[5], 1);
        assert_eq!(table[6], 4);
        assert_eq!(table[7], 2);
        assert_eq!(table[8], 1);
        assert_eq!(table[9], 3);
        assert_eq!(table[10], 4);
        assert_eq!(table[11], 6);
        assert_eq!(table[12], 1);
        assert_eq!(table[13], 1);
        assert_eq!(table[14], 2);
        assert_eq!(table[15], 0);
        assert_eq!(table[16], 1);
        assert_eq!(table[17], 3);
        assert_eq!(table[18], 1);
        assert_eq!(table[19], 1);
        assert_eq!(table[20], 1);
        assert_eq!(table[21], 0);
        assert_eq!(table[22], 0);
        assert_eq!(table[23], 0);
        assert_eq!(table[24], 1);
        assert_eq!(table[25], 0);
        assert_eq!(table[26], 0);
        assert_eq!(table[27], 0);
        assert_eq!(table[28], 0);
        assert_eq!(table[29], 0);
        assert_eq!(table[30], 0);
        assert_eq!(table[31], 0);
    }

    #[test]
    fn test_range_min() {
        let arr: Vec<usize> = vec![1, 3, 4, 8, 6, 1, 4, 2];
        let table: Vec<usize> = build_sparse_table(&arr);

        let min: usize = range_min(&table, arr.len(), 1, 4);
        assert_eq!(min, 3);

        let min: usize = range_min(&table, arr.len(), 1, 5);
        assert_eq!(min, 1);

        let min: usize = range_min(&table, arr.len(), 0, 0);
        assert_eq!(min, 1);
    }

    #[test]
    fn test_fenwick_tree() {
        let arr: Vec<i32> = vec![3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3];
        let fenwick_tree: Vec<i32> = build_fenwick_tree(&arr);
        let sum: i32 = range_sum_fenwick(&fenwick_tree, 5);
        assert_eq!(sum, 15);
    }
}
