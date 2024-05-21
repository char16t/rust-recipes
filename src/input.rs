use std::io;
use std::io::{Read, BufRead};
use std::str::{SplitWhitespace, FromStr};

#[allow(dead_code)]
pub fn take_num<T>() -> T where T: FromStr, <T as FromStr>::Err: std::fmt::Debug {
    _take_num_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_2nums<T1, T2>() -> (T1, T2) 
where 
    T1: FromStr, <T1 as FromStr>::Err: std::fmt::Debug, 
    T2: FromStr, <T2 as FromStr>::Err: std::fmt::Debug
{
    _take_2nums_from_reader(std::io::stdin())
}

fn _take_num_from_reader<T, R>(reader: R) -> T where T: FromStr, <T as FromStr>::Err: std::fmt::Debug, R: Read {
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");
    input.trim().parse().unwrap()
}

fn _take_2nums_from_reader<T1, T2, R>(reader: R) -> (T1, T2)  
where 
    T1: FromStr, <T1 as FromStr>::Err: std::fmt::Debug, 
    T2: FromStr, <T2 as FromStr>::Err: std::fmt::Debug, 
    R: Read
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");

    let mut values: SplitWhitespace = input.trim().split_whitespace();
    let ax: T1 = values.next().unwrap().parse().unwrap();
    let ay: T2 = values.next().unwrap().parse().unwrap();
    (ax, ay)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_num() {
        let input: &str = "42\n";
        let result: i64 = _take_num_from_reader(input.as_bytes());
        assert_eq!(result, 42);
    }

    #[test]
    fn test_take_2nums() {
        let input: &str = "-12 100\n";
        let (x, y): (i64, i64) = _take_2nums_from_reader(input.as_bytes());
        assert_eq!(x, -12);
        assert_eq!(y, 100);
    }
}