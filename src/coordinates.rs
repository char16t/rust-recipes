/// Point (a, b) in (N x M) matrix
#[macro_export]
macro_rules! create_coordinate_function_2d {
    ($n:expr, $m:expr) => {
        |x, y| x * $m + y
    };
}
pub use create_coordinate_function_2d;

/// Point (x, y, z) in (N x M x K) matrix
#[macro_export]
macro_rules! create_coordinate_function_3d {
    ($n:expr, $m:expr, $k:expr) => {
        |x, y, z| x * ($m * $k) + y * $k + z
    };
}
pub use create_coordinate_function_3d;

/// Point (a, b, c, d) in (N x M x K x L) matrix
#[macro_export]
macro_rules! create_coordinate_function_4d {
    ($n:expr, $m:expr, $k:expr, $l:expr) => {
        |a, b, c, d| a * ($m * $k * $l) + b * ($k * $l) + c * $l + d
    };
}
pub use create_coordinate_function_4d;

#[cfg(test)]
mod tests {

    #[test]
    fn test_create_coordinate_function_2d() {
        let coordinate_function = create_coordinate_function_2d!(5, 3);
        let result = coordinate_function(2, 1);
        assert_eq!(result, 7)
    }

    #[test]
    fn test_create_coordinate_function_3d() {
        let coordinate_function = create_coordinate_function_3d!(5, 5, 5);
        let result = coordinate_function(1, 1, 2);
        assert_eq!(result, 32)
    }

    #[test]
    fn test_create_coordinate_function_4d() {
        let coordinate_function = create_coordinate_function_4d!(5, 5, 5, 5);
        let result = coordinate_function(1, 1, 2, 1);
        assert_eq!(result, 161)
    }
}
