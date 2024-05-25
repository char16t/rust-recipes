use crate::create_coordinate_function_2d;

pub fn set_bit(n: isize, pos: usize) -> isize {
    n | (1 << pos)
}

pub fn unset_bit(n: isize, pos: usize) -> isize {
    n & !(1 << pos)
}

pub fn toggle_bit(n: isize, pos: usize) -> isize {
    n ^ (1 << pos)
}

pub fn is_bit_set(n: isize, pos: usize) -> bool {
    (n & (1 << pos)) != 0
}

pub fn is_power_of_two(n: isize) -> bool {
    (n & (n - 1)) == 0
}

/// Brian Kernighan's algorithm
pub fn count_set_bits(n: isize) -> isize {
    let mut count: isize = 0;
    let mut num: isize = n;
    while num != 0 {
        num &= num - 1;
        count += 1;
    }
    count
}

pub fn set_bit_in_matrix(matrix: isize, _n: usize, m: usize, row: usize, col: usize) -> isize {
    let xy = create_coordinate_function_2d!(_n, m);
    set_bit(matrix, xy(row, col))
}

pub fn unset_bit_in_matrix(matrix: isize, _n: usize, m: usize, row: usize, col: usize) -> isize {
    let xy = create_coordinate_function_2d!(_n, m);
    unset_bit(matrix, xy(row, col))
}

pub fn toggle_bit_in_matrix(matrix: isize, _n: usize, m: usize, row: usize, col: usize) -> isize {
    let xy = create_coordinate_function_2d!(_n, m);
    toggle_bit(matrix, xy(row, col))
}

pub fn is_bit_set_in_matrix(matrix: isize, _n: usize, m: usize, row: usize, col: usize) -> bool {
    let xy = create_coordinate_function_2d!(_n, m);
    is_bit_set(matrix, xy(row, col))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_bit() {
        assert_eq!(set_bit(0b0001, 0), 0b0001);
        assert_eq!(set_bit(0b0001, 1), 0b0011);       
    }

    #[test]
    fn test_unset_bit() {
        assert_eq!(unset_bit(0b0001, 0), 0b0000);
        assert_eq!(unset_bit(0b0001, 1), 0b0001);       
    }

    #[test]
    fn test_toggle_bit() {
        assert_eq!(toggle_bit(0b0001, 0), 0b0000);
        assert_eq!(toggle_bit(0b0001, 1), 0b0011);
    }

    #[test]
    fn test_is_bit_set() {
        assert_eq!(is_bit_set(0b0001, 0), true);
        assert_eq!(is_bit_set(0b0001, 1),false);
    }

    #[test]
    fn test_is_power_of_two() {
        assert_eq!(is_power_of_two(0b0000), true);
        assert_eq!(is_power_of_two(0b0001), true);
        assert_eq!(is_power_of_two(0b0010), true);
        assert_eq!(is_power_of_two(0b0100), true);
        assert_eq!(is_power_of_two(0b0110), false);
        assert_eq!(is_power_of_two(0b0101), false);
    }

    #[test]
    fn test_count_set_bits() {
        assert_eq!(count_set_bits(0b0000), 0);
        assert_eq!(count_set_bits(0b1000), 1);
        assert_eq!(count_set_bits(0b1100), 2);
        assert_eq!(count_set_bits(0b1101), 3);
        assert_eq!(count_set_bits(0b1111), 4);
    }

    #[test]
    fn test_set_bit_in_matrix() {
        assert_eq!(set_bit_in_matrix(0b0001, 2, 2, 0, 0), 0b0001);
        assert_eq!(set_bit_in_matrix(0b0001, 2, 2, 1, 1), 0b1001);
    }

    #[test]
    fn test_unset_bit_in_matrix() {
        assert_eq!(unset_bit_in_matrix(0b0001, 2, 2, 0, 0), 0b0000);
        assert_eq!(unset_bit_in_matrix(0b0001, 2, 2, 1, 0), 0b0001);       
    }

    #[test]
    fn test_toggle_bit_in_matrix() {
        assert_eq!(toggle_bit_in_matrix(0b0001, 2, 2, 0, 0), 0b0000);
        assert_eq!(toggle_bit_in_matrix(0b0001, 2, 2, 0, 1), 0b0011);
    }

    #[test]
    fn test_is_bit_set_in_matrix() {
        assert_eq!(is_bit_set_in_matrix(0b0001, 2, 2, 0, 0), true);
        assert_eq!(is_bit_set_in_matrix(0b0001, 2, 2, 0, 1),false);
    }
}