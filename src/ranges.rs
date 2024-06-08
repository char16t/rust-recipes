use std::cmp;
use crate::coordinates;

#[repr(transparent)]
pub struct PrefixSumArray<T> {
    prefix_sum: Vec<T>
}
impl<T> PrefixSumArray<T>
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy
{
    pub fn new(arr: &[T]) -> Self {
        let mut prefix_sum: Vec<T> = vec![T::default(); arr.len()];
        prefix_sum[0] = arr[0];
        for i in 1..prefix_sum.len() {
            prefix_sum[i] = prefix_sum[i - 1] + arr[i];
        }
        Self{ prefix_sum }
    }

    pub fn range_sum(&self, l: usize, r: usize) -> T {
        if l == 0 {
            self.prefix_sum[r]
        } else {
            self.prefix_sum[r] - self.prefix_sum[l-1]
        }
    }

    pub fn len(&self) -> usize {
        self.prefix_sum.len()
    }
}

impl<T> std::ops::Index<usize> for PrefixSumArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.prefix_sum[index]
    }
}

#[allow(dead_code)]
pub struct PrefixSumArray2D<T> {
    prefix_sum: Vec<T>,
    rows: usize, 
    cols: usize
}
impl<T> PrefixSumArray2D<T>
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy
{
    pub fn new(arr: &[T], rows: usize, cols: usize) -> Self {
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

        Self{ prefix_sum, rows, cols }
    }

    pub fn range_sum(&self, top_left: (usize, usize), bottom_right: (usize, usize)) -> T {
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
        let xy = coordinates::create_coordinate_function_2d!(self.rows, self.cols);
        if top_left.0 == 0 && top_left.1 == 0 {
            let a: usize = xy(bottom_right.0, bottom_right.1);
            return self.prefix_sum[a];
        } else if top_left.0 == 0 {
            let a: usize = xy(bottom_right.0, bottom_right.1);
            let b: usize = xy(bottom_right.0, top_left.1 - 1);
            return self.prefix_sum[a] - self.prefix_sum[b];
        } else if top_left.1 == 0 {
            let a: usize = xy(bottom_right.0, bottom_right.1);
            let c: usize = xy(top_left.0 - 1,  bottom_right.1);
            return self.prefix_sum[a] - self.prefix_sum[c];
        } else {
            let a: usize = xy(bottom_right.0, bottom_right.1);
            let b: usize = xy(bottom_right.0, top_left.1 - 1);
            let c: usize = xy(top_left.0 - 1,  bottom_right.1);
            let d: usize = xy(top_left.0 - 1, top_left.1 - 1);
            return self.prefix_sum[a] - self.prefix_sum[b] - self.prefix_sum[c] + self.prefix_sum[d]
        }
    }

    pub fn len(&self) -> usize {
        self.prefix_sum.len()
    }
}

impl<T> std::ops::Index<usize> for PrefixSumArray2D<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.prefix_sum[index]
    }
}

#[allow(dead_code)]
pub struct PrefixSumArray3D<T> {
    prefix_sum: Vec<T>,
    rows: usize, 
    cols: usize,
    depth: usize
}
impl<T> PrefixSumArray3D<T>
where 
    T: Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy
{
    pub fn new(arr: &[T], rows: usize, cols: usize, depth: usize) -> Self {
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
        Self { prefix_sum, cols, rows, depth }
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
    pub fn range_sum(&self, top_left: (usize, usize, usize), bottom_right: (usize, usize, usize)) -> T {
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

        let xy = coordinates::create_coordinate_function_3d!(self.rows, self.cols, self.depth);
        let (x1, y1, z1): (usize, usize, usize) = top_left;
        let (x2, y2, z2): (usize, usize, usize) = bottom_right;
        if x1 == 0 && y1 == 0 && z1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)];
        } else if x1 == 0 && y1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)] - self.prefix_sum[xy(x2, y2, z1-1)]
        } else if x1 == 0 && z1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)] - self.prefix_sum[xy(x2, y1-1, z2)]; 
        } else if y1 == 0 && z1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)] - self.prefix_sum[xy(x1-1, y2, z2)];
        } else if x1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)] 
                - self.prefix_sum[xy(x2, y1-1, z2)] 
                - self.prefix_sum[xy(x2, y2, z1-1)]
                + self.prefix_sum[xy(x2, y1-1, z1-1)];
        } else if y1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)] 
                - self.prefix_sum[xy(x1-1, y2, z2)] 
                - self.prefix_sum[xy(x2, y2, z1-1)]
                + self.prefix_sum[xy(x1-1, y2, z1-1)];
        } else if z1 == 0 {
            return self.prefix_sum[xy(x2, y2, z2)] 
                - self.prefix_sum[xy(x1-1, y2, z2)] 
                - self.prefix_sum[xy(x2, y1-1, z2)] 
                + self.prefix_sum[xy(x1-1, y1-1, z2)];
        } else {
            return self.prefix_sum[xy(x2, y2, z2)] 
                - self.prefix_sum[xy(x1-1, y2, z2)] 
                - self.prefix_sum[xy(x2, y1-1, z2)] 
                - self.prefix_sum[xy(x2, y2, z1-1)]
                + self.prefix_sum[xy(x1-1, y1-1, z2)] 
                + self.prefix_sum[xy(x1-1, y2, z1-1)] 
                + self.prefix_sum[xy(x2, y1-1, z1-1)] 
                - self.prefix_sum[xy(x1-1, y1-1, z1-1)];
        }
    }

    pub fn len(&self) -> usize {
        self.prefix_sum.len()
    }
}

impl<T> std::ops::Index<usize> for PrefixSumArray3D<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.prefix_sum[index]
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
        add_to_fenwick_tree::<T, T>(&mut tree, i+1, arr[i]);
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

pub fn build_segment_tree_sum<T>(arr: &[T]) -> Vec<T>
where T: Default + Copy + std::ops::Add<Output = T>
{
    let n: usize = arr.len();
    let mut tree: Vec<T> = vec![T::default(); 2 * n];
    for i in 0..n {
        tree[n + i] = arr[i];
    }
    for i in (1..n).rev() {
        tree[i] = tree[2 * i] + tree[2 * i + 1];
    }
    tree
}

pub fn query_segment_tree_sum<T>(segment_tree: &[T], left: usize, right: usize) -> T
where
    T: Default + Copy + std::ops::Add<Output = T>
{
    let n: usize = segment_tree.len() / 2;
    let mut sum: T = T::default();
    let mut l: usize = left + n;
    let mut r: usize = right + n;
    while l <= r {
        if l % 2 == 1 {
            sum = sum + segment_tree[l];
            l += 1;
        }
        if r % 2 == 0 {
            sum = sum + segment_tree[r];
            r -= 1;
        }
        l /= 2;
        r /= 2;
    }
    sum
}

pub fn update_segment_tree_sum<T>(segment_tree: &mut [T], index: usize, value: T)
where
    T: Default + Copy + std::ops::Add<Output = T>
{
    let n: usize = segment_tree.len() / 2;
    let mut idx: usize = index + n;
    segment_tree[idx] = value;
    while idx > 1 {
        idx /= 2;
        segment_tree[idx] = segment_tree[2 * idx] + segment_tree[2 * idx + 1];
    }
}

pub struct SegmentTree<T, F> {
    tree: Vec<T>,
    n: usize,
    operation: F,
}
impl<T, F> SegmentTree<T, F>
where
    T: Default + Copy + std::ops::Add<Output = T>,
    F: Fn(T, T) -> T
{
    pub fn new(arr: &[T], operation: F) -> SegmentTree<T, F> {
        let n: usize = arr.len();
        let mut tree: Vec<T> = vec![T::default(); 2 * n];
        for i in 0..n {
            tree[n + i] = arr[i];
        }
        for i in (1..n).rev() {
            tree[i] = operation(tree[2 * i], tree[2 * i + 1]);
        }
        SegmentTree{ tree, n, operation }
    }
    pub fn query(&self, left: usize, right: usize) -> T {
        let mut sum: T = T::default();
        let mut l: usize = left + self.n;
        let mut r: usize = right + self.n;
        while l <= r {
            if l % 2 == 1 {
                sum = sum + self.tree[l];
                l += 1;
            }
            if r % 2 == 0 {
                sum = sum + self.tree[r];
                r -= 1;
            }
            l /= 2;
            r /= 2;
        }
        sum
    }
    pub fn update(&mut self, index: usize, value: T) {
        let mut idx: usize = index + self.n;
        self.tree[idx] = value;
        while idx > 1 {
            idx /= 2;
            self.tree[idx] = (self.operation)(self.tree[2 * idx], self.tree[2 * idx + 1]);
        }
    }
}

pub fn build_difference_array<T>(arr: &[T]) -> Vec<T>
where
    T: Default + Copy + std::ops::Sub<Output = T>
{
    let n: usize = arr.len();
    let mut difference_array: Vec<T> = vec![T::default(); n];
    difference_array[0] = arr[0];
    for i in 1..n {
        difference_array[i] = arr[i] - arr[i-1];
    }
    difference_array
}


pub fn update_difference_array<T>(diff_arr: &mut [T], l: usize, r: usize, val: T)
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
{
    diff_arr[l] = diff_arr[l] + val;
    if r + 1 < diff_arr.len() {
        diff_arr[r + 1] = diff_arr[r + 1] - val;
    }
}

pub fn build_original_by_difference_array<T>(diff_arr: &[T]) -> Vec<T>
where 
    T: Default + Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
{
    let psa: PrefixSumArray<T> = PrefixSumArray::new(&diff_arr);
    return psa.prefix_sum;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prefix_sum_array() {
        let original_array: Vec<i32> = vec![1, 2, 3, 4, 5];
        let prefix_sum_array: PrefixSumArray<i32> = PrefixSumArray::new(&original_array);
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
        let psa: PrefixSumArray<i32> = PrefixSumArray::new(&original_array);
        
        assert_eq!(psa.range_sum(1, 3), 9);
        assert_eq!(psa.range_sum(0, 0), 1);
        assert_eq!(psa.range_sum(1, 1), 2);
        assert_eq!(psa.range_sum(1, 2), 5);
        assert_eq!(psa.range_sum(0, 4), 15);
    }

    #[test]
    fn test_build_prefix_sum_array_2d() {
        let original_array: Vec<i32> = vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        let prefix_sum_array: PrefixSumArray2D<i32> = PrefixSumArray2D::new(&original_array, 4, 5);
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
        let prefix_sum_array: PrefixSumArray2D<i32> = PrefixSumArray2D::new(&original_array, 4, 5);
        assert_eq!(prefix_sum_array.rows, 4);
        assert_eq!(prefix_sum_array.cols, 5);

        let sum: i32 = prefix_sum_array.range_sum((1, 1), (2, 3));
        assert_eq!(sum, 6);

        let sum: i32 = prefix_sum_array.range_sum((0, 0), (2, 3));
        assert_eq!(sum, 12);

        let sum: i32 = prefix_sum_array.range_sum((1, 0), (2, 3));
        assert_eq!(sum, 8);

        let sum: i32 = prefix_sum_array.range_sum((0, 1), (2, 3));
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
        let prefix_sum_array: PrefixSumArray2D<i32> = PrefixSumArray2D::new(&original_array, 4, 5);

        prefix_sum_array.range_sum((2, 3), (1, 1));
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
        let prefix_sum_array: PrefixSumArray2D<i32> = PrefixSumArray2D::new(&original_array, 4, 5);

        prefix_sum_array.range_sum((1, 3), (2, 2));
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
        let prefix_sum_array: PrefixSumArray3D<i32> = PrefixSumArray3D::new(&original_array, 4, 4, 4);
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
        let prefix_sum_array: PrefixSumArray3D<i32> = PrefixSumArray3D::new(&original_array, 4, 4, 4);
        
        let sum: i32 = prefix_sum_array.range_sum((0, 0, 0), (0, 0, 0));
        assert_eq!(sum, 1);

        let sum: i32 = prefix_sum_array.range_sum((0, 0, 1), (0, 0, 3));
        assert_eq!(sum, 3);

        let sum: i32 = prefix_sum_array.range_sum((0, 1, 0), (0, 1, 3));
        assert_eq!(sum, 4);

        let sum: i32 = prefix_sum_array.range_sum((1, 0, 0), (1, 1, 3));
        assert_eq!(sum, 8);

        let sum: i32 = prefix_sum_array.range_sum((1, 1, 0), (1, 1, 3));
        assert_eq!(sum, 4);

        let sum: i32 = prefix_sum_array.range_sum( (1, 0, 1), (2, 0, 3));
        assert_eq!(sum, 6);

        let sum: i32 = prefix_sum_array.range_sum((0, 1, 1), (2, 1, 3));
        assert_eq!(sum, 9);

        let sum: i32 = prefix_sum_array.range_sum((1, 1, 1), (1, 1, 1));
        assert_eq!(sum, 1);

        let sum: i32 = prefix_sum_array.range_sum((1, 1, 1), (3, 3, 3));
        assert_eq!(sum, 27);

        let sum: i32 = prefix_sum_array.range_sum((0, 0, 0), (3, 3, 3));
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
        let prefix_sum_array: PrefixSumArray3D<i32> = PrefixSumArray3D::new(&original_array, 4, 4, 4);
        
        prefix_sum_array.range_sum((1, 0, 0), (0, 0, 0));
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
        let prefix_sum_array: PrefixSumArray3D<i32> = PrefixSumArray3D::new(&original_array, 4, 4, 4);
        
        prefix_sum_array.range_sum((0, 1, 0), (0, 0, 0));
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
        let prefix_sum_array: PrefixSumArray3D<i32> = PrefixSumArray3D::new(&original_array, 4, 4, 4);
        
        prefix_sum_array.range_sum((0, 0, 1), (0, 0, 0));
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

    #[test]
    fn test_segment_tree_sum() {
        let arr: Vec<i32> = vec![1, 3, 5, 7, 9];
        let mut segment_tree: Vec<i32> = build_segment_tree_sum(&arr);

        assert_eq!(segment_tree[0], 0);
        assert_eq!(segment_tree[1], 25);
        assert_eq!(segment_tree[2], 17);
        assert_eq!(segment_tree[3], 8);
        assert_eq!(segment_tree[4], 16);
        assert_eq!(segment_tree[5], 1);
        assert_eq!(segment_tree[6], 3);
        assert_eq!(segment_tree[7], 5);
        assert_eq!(segment_tree[8], 7);
        assert_eq!(segment_tree[9], 9);

        let sum: i32 = query_segment_tree_sum(&segment_tree, 0, 0);
        assert_eq!(sum, 1);

        let sum: i32 = query_segment_tree_sum(&segment_tree, 1, 3);
        assert_eq!(sum, 15);

        update_segment_tree_sum(&mut segment_tree, 2, 10);
        let sum: i32 = query_segment_tree_sum(&segment_tree, 1, 3);
        assert_eq!(sum, 20);
    }

    #[test]
    fn test_segment_tree_struct_sum() {
        let arr: Vec<i32> = vec![1, 3, 5, 7, 9];
        let sum_op = |a, b| { a + b };
        let mut tree = SegmentTree::new(&arr, sum_op);
        
        assert_eq!(tree.query(0, 0), 1);
        assert_eq!(tree.query(1, 3), 15);
        tree.update(2, 10);
        assert_eq!(tree.query(1, 3), 20);
    }

    #[test]
    fn test_segment_tree_struct_logical_ops() {
        let arr: Vec<u8> = vec![0, 0, 0, 1, 1];
        let sum_op = |a, b| { a | b };
        let mut tree = SegmentTree::new(&arr, sum_op);
        
        assert_eq!(tree.query(0, 0), 0);
        assert_eq!(tree.query(1, 3), 1);
        assert_eq!(tree.query(1, 2), 0);
        tree.update(2, 1);
        assert_eq!(tree.query(1, 2), 1);
    }

    #[test]
    fn test_difference_array() {
        let arr: Vec<i32> = vec![3, 1, 4, 1, 5];
        let mut difference_array: Vec<i32> = build_difference_array(&arr);

        assert_eq!(difference_array[0], 3);
        assert_eq!(difference_array[1], -2);
        assert_eq!(difference_array[2], 3);
        assert_eq!(difference_array[3], -3);
        assert_eq!(difference_array[4], 4);

        update_difference_array(&mut difference_array, 1, 3, 5);
        let restored: Vec<i32> = build_original_by_difference_array(&difference_array);
        assert_eq!(restored[0], 3);
        assert_eq!(restored[1], 6);
        assert_eq!(restored[2], 9);
        assert_eq!(restored[3], 6);
        assert_eq!(restored[4], 5);

        update_difference_array(&mut difference_array, 1, 3, -4);
        let restored: Vec<i32> = build_original_by_difference_array(&difference_array);
        assert_eq!(restored[0], 3);
        assert_eq!(restored[1], 2);
        assert_eq!(restored[2], 5);
        assert_eq!(restored[3], 2);
        assert_eq!(restored[4], 5);
    }
}
