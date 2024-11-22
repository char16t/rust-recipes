use std::io;
use std::io::{BufRead, Read};
use std::str::{FromStr, SplitWhitespace};

#[allow(dead_code)]
pub fn take_num<T>() -> T
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
{
    _take_num_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_2nums<T1, T2>() -> (T1, T2)
where
    T1: FromStr,
    <T1 as FromStr>::Err: std::fmt::Debug,
    T2: FromStr,
    <T2 as FromStr>::Err: std::fmt::Debug,
{
    _take_2nums_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_3nums<T1, T2, T3>() -> (T1, T2, T3)
where
    T1: FromStr,
    <T1 as FromStr>::Err: std::fmt::Debug,
    T2: FromStr,
    <T2 as FromStr>::Err: std::fmt::Debug,
    T3: FromStr,
    <T3 as FromStr>::Err: std::fmt::Debug,
{
    _take_3nums_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_4nums<T1, T2, T3, T4>() -> (T1, T2, T3, T4)
where
    T1: FromStr,
    <T1 as FromStr>::Err: std::fmt::Debug,
    T2: FromStr,
    <T2 as FromStr>::Err: std::fmt::Debug,
    T3: FromStr,
    <T3 as FromStr>::Err: std::fmt::Debug,
    T4: FromStr,
    <T4 as FromStr>::Err: std::fmt::Debug,
{
    _take_4nums_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_vector<T>() -> Vec<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
{
    _take_vector_from_reader(std::io::stdin())
}

#[allow(dead_code)]
pub fn take_string<T>() -> Vec<char> {
    _take_string_from_reader(std::io::stdin())
}

fn _take_num_from_reader<T, R>(reader: R) -> T
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
    R: Read,
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");
    input.trim().parse().unwrap()
}

fn _take_2nums_from_reader<T1, T2, R>(reader: R) -> (T1, T2)
where
    T1: FromStr,
    <T1 as FromStr>::Err: std::fmt::Debug,
    T2: FromStr,
    <T2 as FromStr>::Err: std::fmt::Debug,
    R: Read,
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");

    let mut values: SplitWhitespace = input.split_whitespace();
    let ax: T1 = values.next().unwrap().parse().unwrap();
    let ay: T2 = values.next().unwrap().parse().unwrap();
    (ax, ay)
}

fn _take_3nums_from_reader<T1, T2, T3, R>(reader: R) -> (T1, T2, T3)
where
    T1: FromStr,
    <T1 as FromStr>::Err: std::fmt::Debug,
    T2: FromStr,
    <T2 as FromStr>::Err: std::fmt::Debug,
    T3: FromStr,
    <T3 as FromStr>::Err: std::fmt::Debug,
    R: Read,
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");

    let mut values: SplitWhitespace = input.split_whitespace();
    let a: T1 = values.next().unwrap().parse().unwrap();
    let b: T2 = values.next().unwrap().parse().unwrap();
    let c: T3 = values.next().unwrap().parse().unwrap();
    (a, b, c)
}

fn _take_4nums_from_reader<T1, T2, T3, T4, R>(reader: R) -> (T1, T2, T3, T4)
where
    T1: FromStr,
    <T1 as FromStr>::Err: std::fmt::Debug,
    T2: FromStr,
    <T2 as FromStr>::Err: std::fmt::Debug,
    T3: FromStr,
    <T3 as FromStr>::Err: std::fmt::Debug,
    T4: FromStr,
    <T4 as FromStr>::Err: std::fmt::Debug,
    R: Read,
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");

    let mut values: SplitWhitespace = input.split_whitespace();
    let a: T1 = values.next().unwrap().parse().unwrap();
    let b: T2 = values.next().unwrap().parse().unwrap();
    let c: T3 = values.next().unwrap().parse().unwrap();
    let d: T4 = values.next().unwrap().parse().unwrap();
    (a, b, c, d)
}

fn _take_vector_from_reader<T, R>(reader: R) -> Vec<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
    R: Read,
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).unwrap();
    let arr: Vec<T> = input
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    arr
}

fn _take_string_from_reader<R>(reader: R) -> Vec<char>
where
    R: Read,
{
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).unwrap();
    let vec: Vec<char> = input.trim().chars().collect();
    vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{benchmark, numbers};

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

    #[test]
    fn test_take_3nums() {
        let input: &str = "-12 100 15\n";
        let (x, y, z): (i64, i64, i64) = _take_3nums_from_reader(input.as_bytes());
        assert_eq!(x, -12);
        assert_eq!(y, 100);
        assert_eq!(z, 15);
    }

    #[test]
    fn test_take_4nums() {
        let input: &str = "-12 100 15 3454564643\n";
        let (a, b, c, d): (i64, i64, i64, i64) = _take_4nums_from_reader(input.as_bytes());
        assert_eq!(a, -12);
        assert_eq!(b, 100);
        assert_eq!(c, 15);
        assert_eq!(d, 3454564643);
    }

    #[test]
    fn test_take_vector() {
        let input: &str = "-12 100 15 3454564643\n";
        let vec: Vec<i64> = _take_vector_from_reader(input.as_bytes());
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0], -12);
        assert_eq!(vec[1], 100);
        assert_eq!(vec[2], 15);
        assert_eq!(vec[3], 3454564643);
    }

    #[test]
    fn test_take_string() {
        let input: &str = "abcdef\n";
        let vec: Vec<char> = _take_string_from_reader(input.as_bytes());
        assert_eq!(vec.len(), 6);
        assert_eq!(vec[0], 'a');
        assert_eq!(vec[1], 'b');
        assert_eq!(vec[2], 'c');
        assert_eq!(vec[3], 'd');
        assert_eq!(vec[4], 'e');
        assert_eq!(vec[5], 'f');
    }

    #[test]
    fn bench_take_num() {
        let (_result, runtime_secs): ((), f64) = benchmark::time(test_take_num);
        println!("Runtime (seconds): {}", runtime_secs);
        assert!(numbers::approx_equal(runtime_secs, 0.000024115, 0.001));
    }
}
