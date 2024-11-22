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

pub fn sets_union(a: isize, b: isize) -> isize {
    a | b
}

pub fn sets_intersection(a: isize, b: isize) -> isize {
    a & b
}

pub fn sets_symetric_difference(a: isize, b: isize) -> isize {
    a ^ b
}

pub fn sets_difference(a: isize, b: isize) -> isize {
    a & !b
}

pub fn sets_is_subset(a: isize, b: isize) -> bool {
    (a & b) == a
}

pub fn sets_is_superset(a: isize, b: isize) -> bool {
    (a & b) == b
}

const BITS_PER_U64: usize = 64;

pub struct LongArithmetic {
    data: Vec<u64>,
}

impl LongArithmetic {
    pub fn new() -> Self {
        LongArithmetic {
            data: Vec::with_capacity(1),
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        let cap: usize = size / BITS_PER_U64 + 1;
        let mut a: LongArithmetic = LongArithmetic {
            data: Vec::with_capacity(1),
        };
        a.data.resize_with(cap, Default::default);
        a
    }

    pub fn set_bit(&mut self, index: usize) {
        let array_index: usize = index / BITS_PER_U64;
        let bit_index: usize = index % BITS_PER_U64;

        while array_index >= self.data.len() {
            self.data.push(0);
        }

        self.data[array_index] |= 1 << bit_index;
    }

    pub fn unset_bit(&mut self, index: usize) {
        let array_index: usize = index / BITS_PER_U64;
        let bit_index: usize = index % BITS_PER_U64;

        while array_index >= self.data.len() {
            self.data.push(0);
        }

        self.data[array_index] &= !(1 << bit_index);
    }

    pub fn toggle_bit(&mut self, index: usize) {
        let array_index: usize = index / BITS_PER_U64;
        let bit_index: usize = index % BITS_PER_U64;

        while array_index >= self.data.len() {
            self.data.push(0);
        }

        self.data[array_index] ^= 1 << bit_index;
    }

    pub fn is_bit_set(&mut self, index: usize) -> bool {
        let array_index: usize = index / BITS_PER_U64;
        let bit_index: usize = index % BITS_PER_U64;

        while array_index >= self.data.len() {
            self.data.push(0);
        }

        (self.data[array_index] & (1 << bit_index)) != 0
    }
}

#[allow(unused_must_use)]
impl std::fmt::Debug for LongArithmetic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, &num) in self.data.iter().enumerate() {
            write!(f, "Array[{}]: {:064b} ({})\n", i, num, num);
        }
        Ok(())
    }
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
        assert_eq!(is_bit_set(0b0001, 1), false);
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
        assert_eq!(is_bit_set_in_matrix(0b0001, 2, 2, 0, 1), false);
    }

    #[test]
    fn test_sets_union() {
        assert_eq!(sets_union(0b10101, 0b01010), 0b11111);
        assert_eq!(sets_union(0b10111, 0b01010), 0b11111);
    }

    #[test]
    fn test_sets_intersection() {
        assert_eq!(sets_intersection(0b10101, 0b01010), 0b00000);
        assert_eq!(sets_intersection(0b10111, 0b01010), 0b00010);
    }

    #[test]
    fn test_sets_symetric_difference() {
        assert_eq!(sets_symetric_difference(0b10101, 0b01010), 0b11111);
        assert_eq!(sets_symetric_difference(0b10111, 0b01010), 0b11101);
    }

    #[test]
    fn test_sets_difference() {
        assert_eq!(sets_difference(0b10101, 0b01010), 0b10101);
        assert_eq!(sets_difference(0b11111, 0b11010), 0b00101);
    }

    #[test]
    fn test_sets_is_subset() {
        assert_eq!(sets_is_subset(0b10101, 0b01010), false);
        assert_eq!(sets_is_subset(0b11111, 0b11010), false);

        assert_eq!(sets_is_subset(0b00010, 0b11111), true);
    }

    #[test]
    fn test_sets_is_superset() {
        assert_eq!(sets_is_superset(0b11111, 0b11010), true);
        assert_eq!(sets_is_superset(0b00010, 0b11111), false);
    }

    #[test]
    fn test_long_arithmetic_with_capacity() {
        let a: LongArithmetic = LongArithmetic::with_capacity(64);
        let expected_output: &str = concat!(
            "Array[0]: 0000000000000000000000000000000000000000000000000000000000000000 (0)\n",
            "Array[1]: 0000000000000000000000000000000000000000000000000000000000000000 (0)\n"
        );
        assert_eq!(format!("{:?}", a), expected_output);
    }

    #[test]
    fn test_long_arithmetic_debug() {
        let mut a: LongArithmetic = LongArithmetic::new();
        a.set_bit(65);
        let expected_output: &str = concat!(
            "Array[0]: 0000000000000000000000000000000000000000000000000000000000000000 (0)\n",
            "Array[1]: 0000000000000000000000000000000000000000000000000000000000000010 (2)\n"
        );
        assert_eq!(format!("{:?}", a), expected_output);
    }

    #[test]
    fn test_long_arithmetic_set_bit() {
        let mut a: LongArithmetic = LongArithmetic::new();
        a.set_bit(65);
        assert_eq!(a.is_bit_set(65), true);
        assert_eq!(a.is_bit_set(128), false);
    }

    #[test]
    fn test_long_arithmetic_unset_bit() {
        let mut a: LongArithmetic = LongArithmetic::new();
        a.set_bit(65);
        a.unset_bit(65);
        assert_eq!(a.is_bit_set(65), false);

        a.unset_bit(256);
        assert_eq!(a.is_bit_set(256), false);
    }

    #[test]
    fn test_long_arithmetic_toggle_bit() {
        let mut a: LongArithmetic = LongArithmetic::new();
        a.toggle_bit(65);
        assert_eq!(a.is_bit_set(65), true);
        a.toggle_bit(65);
        assert_eq!(a.is_bit_set(65), false);
        a.toggle_bit(65);
        assert_eq!(a.is_bit_set(65), true);
    }
}
