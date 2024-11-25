use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::Sub;

#[allow(dead_code)]
#[derive(Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}
impl<T> Matrix<T>
where
    T: Default + Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }
    pub fn new_vector(size: usize) -> Self {
        Self {
            rows: size,
            cols: 1,
            data: vec![T::default(); size],
        }
    }
    #[allow(clippy::manual_memcpy)]
    pub fn from_vector(vec: &[T]) -> Self {
        let mut data: Vec<T> = vec![T::default(); vec.len()];
        for i in 0..vec.len() {
            data[i] = vec[i];
        }
        Self {
            rows: vec.len(),
            cols: 1,
            data,
        }
    }
    pub fn new_diag(size: usize, elem: T) -> Self {
        let mut data: Vec<T> = vec![T::default(); size * size];
        for i in 0..size {
            data[i * size + i] = elem;
        }
        Self {
            rows: size,
            cols: size,
            data,
        }
    }
    pub fn transpose(&self) -> Self {
        let mut transposed: Matrix<T> = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        transposed
    }
    pub fn pow(&self, n: usize) -> Self {
        if self.rows != self.cols {
            panic!("Matrix exponentiation available only for square matrices");
        }
        if n == 0 {
            panic!("Raising the matrix to the zero degree is not implemented. Raising the matrix to the power of zero returns a unit of the same size.");
        }

        let mut result: Matrix<T> = self.clone();
        let mut base: Matrix<T> = self.clone();
        let mut exp: usize = n - 1;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.clone() * base.clone();
            }
            base = base.clone() * base;
            exp /= 2;
        }

        result
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start: usize = index * self.cols;
        &self.data[start..start + self.cols]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start: usize = index * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

impl<T> Add for Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Matrix<T> {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for addition");
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for substraction");
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Mul for Matrix<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        if self.cols != other.rows {
            panic!(
                "First matrix [{} x {}] rows number must match with second matrix [{} x {}] cols number for multiplication",
                self.rows, self.cols,
                other.rows, other.cols
            );
        }

        let mut result_data: Vec<T> = vec![T::default(); self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result_data[i * other.cols + j] = result_data[i * other.cols + j]
                        + self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        }
    }
}

impl<T> Mul<T> for Matrix<T>
where
    T: Clone + std::ops::Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, scalar: T) -> Matrix<T> {
        let result_data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }
}

impl<T> std::fmt::Display for Matrix<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:.2} ", self.data[i * self.cols + j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn fib(n: usize) -> u128 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut m: Matrix<u128> = Matrix::new(2, 2);
    m[0][0] = 0;
    m[0][1] = 1;
    m[1][0] = 1;
    m[1][1] = 1;

    let mut v: Matrix<u128> = Matrix::new_vector(2);
    v[0][0] = 0;
    v[1][0] = 1;

    let r: Matrix<u128> = m.pow(n) * v;

    r[0][0]
}

pub fn linear_recurrent_sequence(coefficients: &[i128], initial: &[i128], n: usize) -> i128 {
    let k: usize = coefficients.len();

    let mut m: Matrix<i128> = Matrix::new(k, k);
    for i in 0..k - 1 {
        for j in 0..k {
            if i + 1 == j {
                m[i][j] = 1;
            }
        }
    }
    for i in 0..k {
        m[k - 1][i] = coefficients[k - i - 1];
    }

    let mut v: Matrix<i128> = Matrix::new_vector(k);
    for i in 0..k {
        v[i][0] = initial[i];
    }

    let r: Matrix<i128> = m.pow(n) * v;
    r[0][0]
}

/// Number of paths that start in A, end in B and consist of exactly N edges.
/// Matrix is a adjacency matrix of the unweighted graph.
pub fn number_of_paths_with_n_egdes(
    a: usize,
    b: usize,
    n: usize,
    adjacency_matrix: Matrix<usize>,
) -> usize {
    adjacency_matrix.pow(n)[a][b]
}

/// Shortest path length that start in A, end in B and consist of exactly N edges.
/// Matrix is a adjacency matrix of the weighted graph.
pub fn shortest_path_with_n_eges(
    a: usize,
    b: usize,
    n: usize,
    adjacency_matrix: Matrix<usize>,
) -> usize {
    fn mul(first: Matrix<usize>, other: Matrix<usize>) -> Matrix<usize> {
        if first.cols != first.rows {
            panic!("First matrix rows number must match with second matrix cols number for multiplication");
        }

        let mut result_data: Vec<usize> = vec![0; first.rows * other.cols];

        for i in 0..first.rows {
            for j in 0..other.cols {
                for k in 0..first.cols {
                    let current: usize = result_data[i * other.cols + j];
                    let next: usize =
                        first.data[i * first.cols + k] * other.data[k * other.cols + j];
                    if current == 0 {
                        result_data[i * other.cols + j] = next;
                    } else if next == 0 {
                        result_data[i * other.cols + j] = current;
                    } else {
                        result_data[i * other.cols + j] = current.min(next);
                    }
                    // result_data[i * other.cols + j] =
                    //     result_data[i * other.cols + j].min(first.data[i * first.cols + k] * other.data[k * other.cols + j]);
                }
            }
        }

        Matrix {
            rows: first.rows,
            cols: other.cols,
            data: result_data,
        }
    }
    fn pow(m: Matrix<usize>, n: usize) -> Matrix<usize> {
        if m.rows != m.cols {
            panic!("Matrix exponentiation available only for square matrices");
        }
        if n == 0 {
            panic!("Raising the matrix to the zero degree is not implemented. Raising the matrix to the power of zero returns a unit of the same size.");
        }

        let mut result: Matrix<usize> = m.clone();
        let mut base: Matrix<usize> = m.clone();
        let mut exp: usize = n - 1;

        while exp > 0 {
            if exp % 2 == 1 {
                result = mul(result.clone(), base.clone());
            }
            base = mul(base.clone(), base);
            exp /= 2;
        }

        result
    }
    pow(adjacency_matrix, n)[a][b]
}

pub fn gaussian_elimination(system: &mut Matrix<f32>) -> Option<Vec<f32>> {
    fn swap_rows(system: &mut Matrix<f32>, a: usize, b: usize) {
        if a == b {
            return;
        }
        let length: usize = system.cols;
        for i in 0..length {
            let tmp: f32 = system[a][i];
            system[a][i] = system[b][i];
            system[b][i] = tmp;
        }
    }

    // echelon form
    for i in 0..system.rows {
        if system[i][i] == 0.0 {
            let mut swap_with: usize = i;
            for l in i + 1..system.rows {
                if system[l][i] != 0f32 {
                    swap_with = l;
                    break;
                }
            }
            swap_rows(system, i, swap_with);
        }
        for j in i + 1..system.rows {
            let factor = system[j][i] / system[i][i];
            for k in i..system.cols {
                system[j][k] -= factor * system[i][k];
            }
        }
    }

    // Process follows a similar pattern but this one reduces the upper triangular elements
    for i in (1..system.rows).rev() {
        if system[i][i] == 0f32 {
            let mut swap_with: usize = i;
            for l in (1..i + 1).rev() {
                if system[l][i] != 0f32 {
                    swap_with = l;
                    break;
                }
            }
            swap_rows(system, i, swap_with);
        }
        for j in (1..i + 1).rev() {
            let factor = system[j - 1][i] / system[i][i];
            for k in (0..system.cols).rev() {
                system[j - 1][k] -= factor * system[i][k];
            }
        }
    }

    // produce solutions through back substitution
    let mut solutions: Vec<f32> = vec![];
    let n: usize = system.rows;
    for i in 0..n {
        if system[i][i] == 0f32 {
            return None;
        } else {
            system[i][n] /= system[i][i];
            system[i][i] = 1f32;
            solutions.push(system[i][n])
        }
    }
    Some(solutions)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_matrix() {
        let m: Matrix<usize> = Matrix::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data[0], 0);
        assert_eq!(m.data[1], 0);
        assert_eq!(m.data[2], 0);
        assert_eq!(m.data[3], 0);
        assert_eq!(m.data[4], 0);
        assert_eq!(m.data[5], 0);
    }

    #[test]
    fn test_create_vector() {
        let m: Matrix<usize> = Matrix::new_vector(3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 1);
        assert_eq!(m.data[0], 0);
        assert_eq!(m.data[1], 0);
        assert_eq!(m.data[2], 0);

        let m: Matrix<usize> = Matrix::from_vector(&[10, 20, 30]);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 1);
        assert_eq!(m.data[0], 10);
        assert_eq!(m.data[1], 20);
        assert_eq!(m.data[2], 30);
    }

    #[test]
    fn test_matrix_diag() {
        let m: Matrix<usize> = Matrix::new_diag(3, 1);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(m[0][0], 1);
        assert_eq!(m[0][1], 0);
        assert_eq!(m[0][2], 0);
        assert_eq!(m[1][0], 0);
        assert_eq!(m[1][1], 1);
        assert_eq!(m[1][2], 0);
        assert_eq!(m[2][0], 0);
        assert_eq!(m[2][1], 0);
        assert_eq!(m[2][2], 1);

        let m: Matrix<usize> = Matrix::new_diag(1, 1);
        assert_eq!(m[0][0], 1);
    }

    #[test]
    fn test_matrix_data_access() {
        let mut m: Matrix<usize> = Matrix::new(2, 3);
        m[0][0] = 1;
        m[0][1] = 2;
        m[0][2] = 3;
        m[1][0] = 4;
        m[1][1] = 5;
        m[1][2] = 6;
        assert_eq!(m.data, vec![1, 2, 3, 4, 5, 6]);

        assert_eq!(m[0][0], 1);
        assert_eq!(m[0][0], 1);
        assert_eq!(m[0][1], 2);
        assert_eq!(m[0][2], 3);
        assert_eq!(m[1][0], 4);
        assert_eq!(m[1][1], 5);
        assert_eq!(m[1][2], 6);
    }

    #[test]
    fn test_matrix_transpose() {
        // 1 2 3
        // 4 5 6
        let mut m: Matrix<usize> = Matrix::new(2, 3);
        m[0][0] = 1;
        m[0][1] = 2;
        m[0][2] = 3;
        m[1][0] = 4;
        m[1][1] = 5;
        m[1][2] = 6;

        // 1 4
        // 2 5
        // 3 6
        let transposed: Matrix<usize> = m.transpose();
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.data, vec![1, 4, 2, 5, 3, 6]);

        assert_eq!(transposed[0][0], 1);
        assert_eq!(transposed[0][1], 4);
        assert_eq!(transposed[1][0], 2);
        assert_eq!(transposed[1][1], 5);
        assert_eq!(transposed[2][0], 3);
        assert_eq!(transposed[2][1], 6);
    }

    #[test]
    fn test_matrix_add() {
        let a: Matrix<i32> = Matrix::new(2, 2);
        let b: Matrix<i32> = Matrix::new(2, 2);
        let c: Matrix<i32> = a + b;
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.data, vec![0, 0, 0, 0]);

        let mut a: Matrix<i32> = Matrix::new(2, 2);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let mut b: Matrix<i32> = Matrix::new(2, 2);
        b[0][0] = 5;
        b[0][1] = 6;
        b[1][0] = 7;
        b[1][1] = 8;

        let c: Matrix<i32> = a + b;
        assert_eq!(c[0][0], 6);
        assert_eq!(c[0][1], 8);
        assert_eq!(c[1][0], 10);
        assert_eq!(c[1][1], 12);
    }

    #[test]
    #[should_panic]
    fn test_matrix_add_panic() {
        let a: Matrix<i32> = Matrix::new(3, 2);
        let b: Matrix<i32> = Matrix::new(2, 3);
        let _: Matrix<i32> = a + b;
    }

    #[test]
    fn test_matrix_sub() {
        let a: Matrix<i32> = Matrix::new(2, 2);
        let b: Matrix<i32> = Matrix::new(2, 2);
        let c: Matrix<i32> = a + b;
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.data, vec![0, 0, 0, 0]);

        let mut a: Matrix<i32> = Matrix::new(2, 2);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let mut b: Matrix<i32> = Matrix::new(2, 2);
        b[0][0] = 5;
        b[0][1] = 6;
        b[1][0] = 7;
        b[1][1] = 8;

        let c: Matrix<i32> = a - b;
        assert_eq!(c[0][0], -4);
        assert_eq!(c[0][1], -4);
        assert_eq!(c[1][0], -4);
        assert_eq!(c[1][1], -4);
    }

    #[test]
    #[should_panic]
    fn test_matrix_sub_panic() {
        let a: Matrix<i32> = Matrix::new(3, 2);
        let b: Matrix<i32> = Matrix::new(2, 3);
        let _: Matrix<i32> = a - b;
    }

    #[test]
    fn test_matrix_mul() {
        let mut a: Matrix<i32> = Matrix::new(2, 3);
        a[0][0] = 1;
        a[0][1] = 2;
        a[0][2] = 3;
        a[1][0] = 4;
        a[1][1] = 5;
        a[1][2] = 6;

        let mut b: Matrix<i32> = Matrix::new(3, 2);
        b[0][0] = 7;
        b[0][1] = 8;
        b[1][0] = 9;
        b[1][1] = 10;
        b[2][0] = 11;
        b[2][1] = 12;

        let c: Matrix<i32> = a * b;
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c[0][0], 58);
        assert_eq!(c[0][1], 64);
        assert_eq!(c[1][0], 139);
        assert_eq!(c[1][1], 154);
    }

    #[test]
    #[should_panic]
    fn test_matrix_mul_panic() {
        let a: Matrix<i32> = Matrix::new(2, 3);
        let b: Matrix<i32> = Matrix::new(2, 3);
        let _ = a * b;
    }

    #[test]
    fn test_matrix_mul_scalar() {
        let mut a: Matrix<i32> = Matrix::new(2, 3);
        a[0][0] = 1;
        a[0][1] = 2;
        a[0][2] = 3;
        a[1][0] = 4;
        a[1][1] = 5;
        a[1][2] = 6;

        let b: Matrix<i32> = a * 3;
        assert_eq!(b[0][0], 3);
        assert_eq!(b[0][1], 6);
        assert_eq!(b[0][2], 9);
        assert_eq!(b[1][0], 12);
        assert_eq!(b[1][1], 15);
        assert_eq!(b[1][2], 18);
    }

    #[test]
    fn test_matrix_display() {
        let mut a: Matrix<i32> = Matrix::new(2, 2);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;
        let expected_output: &str = "1 2 \n3 4 \n";
        let actual_output: String = format!("{}", a);
        assert_eq!(expected_output, actual_output);
    }

    #[test]
    #[should_panic]
    fn test_matrix_pow_non_square() {
        let a: Matrix<i32> = Matrix::new(2, 3);
        a.pow(2);
    }

    #[test]
    #[should_panic]
    fn test_matrix_pow_zero() {
        let a: Matrix<i32> = Matrix::new(2, 2);
        a.pow(0);
    }

    #[test]
    fn test_matrix_pow() {
        let mut a: Matrix<i32> = Matrix::new(2, 2);
        a[0][0] = 2;
        a[0][1] = 5;
        a[1][0] = 1;
        a[1][1] = 4;

        let b: Matrix<i32> = a.pow(3);
        assert_eq!(b[0][0], 48);
        assert_eq!(b[0][1], 165);
        assert_eq!(b[1][0], 33);
        assert_eq!(b[1][1], 114);

        let c: Matrix<i32> = a.pow(5);
        assert_eq!(c[0][0], 1422);
        assert_eq!(c[0][1], 4905);
        assert_eq!(c[1][0], 981);
        assert_eq!(c[1][1], 3384);

        let d: Matrix<i32> = a.pow(4);
        assert_eq!(d[0][0], 261);
        assert_eq!(d[0][1], 900);
        assert_eq!(d[1][0], 180);
        assert_eq!(d[1][1], 621);

        let e: Matrix<i32> = a.pow(1);
        assert_eq!(e[0][0], 2);
        assert_eq!(e[0][1], 5);
        assert_eq!(e[1][0], 1);
        assert_eq!(e[1][1], 4);
    }

    #[test]
    fn test_fib() {
        assert_eq!(fib(0), 0);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(2), 1);
        assert_eq!(fib(3), 2);
        assert_eq!(fib(10), 55);
        assert_eq!(fib(14), 377);
        assert_eq!(fib(100), 354224848179261915075);
    }

    #[test]
    fn test_linear_recurrent_sequence() {
        assert_eq!(linear_recurrent_sequence(&[1, 1], &[0, 1], 1), 1);
        assert_eq!(linear_recurrent_sequence(&[1, 1], &[0, 1], 3), 2);
        assert_eq!(linear_recurrent_sequence(&[1, 1], &[0, 1], 9), 34);
        assert_eq!(linear_recurrent_sequence(&[1, 1], &[0, 1], 10), 55);
        assert_eq!(linear_recurrent_sequence(&[1, 1], &[0, 1], 14), 377);
    }

    #[test]
    fn test_number_of_paths_with_n_egdes() {
        let mut m: Matrix<usize> = Matrix::new(6, 6);
        m[0][0] = 0;
        m[0][1] = 0;
        m[0][2] = 0;
        m[0][3] = 1;
        m[0][4] = 0;
        m[0][5] = 0;
        m[1][0] = 1;
        m[1][1] = 0;
        m[1][2] = 0;
        m[1][3] = 0;
        m[1][4] = 1;
        m[1][5] = 1;
        m[2][0] = 0;
        m[2][1] = 1;
        m[2][2] = 0;
        m[2][3] = 0;
        m[2][4] = 0;
        m[2][5] = 0;
        m[3][0] = 0;
        m[3][1] = 1;
        m[3][2] = 0;
        m[3][3] = 0;
        m[3][4] = 0;
        m[3][5] = 0;
        m[4][0] = 0;
        m[4][1] = 0;
        m[4][2] = 0;
        m[4][3] = 0;
        m[4][4] = 0;
        m[4][5] = 0;
        m[5][0] = 0;
        m[5][1] = 0;
        m[5][2] = 1;
        m[5][3] = 0;
        m[5][4] = 1;
        m[5][5] = 0;

        let r: usize = number_of_paths_with_n_egdes(1, 4, 4, m);
        assert_eq!(r, 2);
    }

    #[test]
    fn test_shortest_path_with_n_eges() {
        let mut m: Matrix<usize> = Matrix::new(6, 6);
        m[0][0] = 0;
        m[0][1] = 0;
        m[0][2] = 0;
        m[0][3] = 4;
        m[0][4] = 0;
        m[0][5] = 0;
        m[1][0] = 2;
        m[1][1] = 0;
        m[1][2] = 0;
        m[1][3] = 0;
        m[1][4] = 1;
        m[1][5] = 2;
        m[2][0] = 0;
        m[2][1] = 4;
        m[2][2] = 0;
        m[2][3] = 0;
        m[2][4] = 0;
        m[2][5] = 0;
        m[3][0] = 0;
        m[3][1] = 1;
        m[3][2] = 0;
        m[3][3] = 0;
        m[3][4] = 0;
        m[3][5] = 0;
        m[4][0] = 0;
        m[4][1] = 0;
        m[4][2] = 0;
        m[4][3] = 0;
        m[4][4] = 0;
        m[4][5] = 0;
        m[5][0] = 0;
        m[5][1] = 0;
        m[5][2] = 3;
        m[5][3] = 0;
        m[5][4] = 2;
        m[5][5] = 0;

        let r: usize = shortest_path_with_n_eges(1, 4, 4, m);
        assert_eq!(r, 8);
    }

    #[test]
    fn test_gaussian_elimination() {
        let mut m: Matrix<f32> = Matrix::new(3, 4);
        m[0][0] = 2.0;
        m[0][1] = 4.0;
        m[0][2] = 1.0;
        m[0][3] = 16.0;
        m[1][0] = 1.0;
        m[1][1] = 2.0;
        m[1][2] = 5.0;
        m[1][3] = 17.0;
        m[2][0] = 3.0;
        m[2][1] = 1.0;
        m[2][2] = 1.0;
        m[2][3] = 8.0;
        let solution: Option<Vec<f32>> = gaussian_elimination(&mut m);
        assert_eq!(solution, Some(vec![1.0, 3.0, 2.0]));

        let mut m: Matrix<f32> = Matrix::new(2, 3);
        m[0][0] = 1.0;
        m[0][1] = 1.0;
        m[0][2] = 2.0;
        m[1][0] = 2.0;
        m[1][1] = 2.0;
        m[1][2] = 4.0;
        let solution: Option<Vec<f32>> = gaussian_elimination(&mut m);
        assert_eq!(solution, None);
    }
}
