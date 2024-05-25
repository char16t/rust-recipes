pub fn set_bit(n: isize, pos: usize) -> isize {
    n | (1 << pos)
}

pub fn unset_bit(n: isize, pos: usize) -> isize {
    n & !(1 << pos)
}

pub fn toggle_bit(n: isize, pos: usize) -> isize {
    n ^ (1 << pos)
}

pub fn is_bit_set(n: isize, pos: isize) -> bool {
    (n & (1 << pos)) != 0
}

pub fn is_power_of_two(n: isize) -> bool {
    (n & (n - 1)) == 0
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
}