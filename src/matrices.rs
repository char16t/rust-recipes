use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

#[allow(dead_code)]
#[derive(Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>
}
impl<T> Matrix<T>
where
    T: Default + Copy + Add<Output = T> + Mul<Output = T>
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows, cols, data: vec![T::default(); rows * cols]
        }
    }
    pub fn new_vector(size: usize) -> Self {
        Self {
            rows: size, 
            cols: 1, 
            data: vec![T::default(); size]
        }
    }
    pub fn from_vector(vec: &[T]) -> Self {
        let mut data: Vec<T> = vec![T::default(); vec.len()];
        for i in 0..vec.len() {
            data[i] = vec[i];
        }
        Self {
            rows: vec.len(),
            cols: 1, 
            data
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
            data
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
    T: Add<Output = T> + Clone
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Matrix<T> {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for addition");
        }

        let data: Vec<T> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T> + Clone
{
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for substraction");
        }

        let data: Vec<T> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data
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
    T: std::fmt::Display
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

        let m: Matrix<usize> = Matrix::from_vector(&vec![10, 20, 30]);
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
        a[0][0] = 1; a[0][1] = 2;
        a[1][0] = 3; a[1][1] = 4;
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
}
