use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Add;
use std::ops::Mul;

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
}
