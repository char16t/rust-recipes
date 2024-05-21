use std::io;
use std::io::BufRead;
use std::str::SplitWhitespace;

#[allow(dead_code)]
pub fn take_int() -> i64 {
    _take_int_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_2int() -> (i64, i64) {
    _take_2int_from_reader(std::io::stdin())
}

fn _take_int_from_reader<R: io::Read>(reader: R) -> i64 {
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");
    input.trim().parse().unwrap()
}

fn _take_2int_from_reader<R: io::Read>(reader: R) -> (i64, i64) {
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");

    let mut values: SplitWhitespace = input.trim().split_whitespace();
    let ax: i64 = values.next().unwrap().parse().unwrap();
    let ay: i64 = values.next().unwrap().parse().unwrap();
    (ax, ay)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_int() {
        let input: &str = "42\n";
        let result: i64 = _take_int_from_reader(input.as_bytes());
        assert_eq!(result, 42);
    }

    #[test]
    fn tcest_take_2int() {
        let input: &str = "-12 100\n";
        let (x, y): (i64, i64) = _take_2int_from_reader(input.as_bytes());
        assert_eq!(x, -12);
        assert_eq!(y, 100);
    }
}