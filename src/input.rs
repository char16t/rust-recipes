use std::io;
use std::io::BufRead;

pub fn take_int() -> usize {
    take_int_from_reader(std::io::stdin())
}

fn take_int_from_reader<R: io::Read>(reader: R) -> usize {
    let mut input: String = String::new();
    let mut reader: io::BufReader<R> = io::BufReader::new(reader);
    reader.read_line(&mut input).expect("Failed to read line");
    input.trim().parse().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_int() {
        // Simulate input for testing
        let input: &str = "42\n";
        let result: usize = take_int_from_reader(input.as_bytes());
    
        // Check the output
        assert_eq!(result, 42);
    }
}