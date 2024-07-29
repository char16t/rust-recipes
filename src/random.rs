use std::io::{Read, Result};

pub struct Xoshiro256 {
    s: [u64; 4],
}
impl Xoshiro256 {
    pub fn new() -> Self {
        match get_random_seed() {
            Ok(seed) => Self { s: convert_to_u64_array(seed) },
            Err(err) => panic!("Unable to read entropy from /dev/urandom: {}", err)
        }
    }
    pub fn from_seed(seed: [u64; 4]) -> Self {
        Xoshiro256 { s: seed }
    }
    pub fn rand(&mut self) -> u64 {
        let result: u64 = rol64(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t: u64 = self.s[1] << 17;
    
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
    
        self.s[2] ^= t;
        self.s[3] = rol64(self.s[3], 45);
    
        result
    }
    pub fn rand_float(&mut self) -> f64 {
        (self.rand() as f64) / (u64::MAX as f64)
    }
    pub fn rand_range(&mut self, a: u64, b: u64) -> u64 {
        let m: u64 = b - a + 1;
        return a + (self.rand() % m);
    }
    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        if a.len() == 0 {
            return;
        }
        let mut i: usize = a.len() - 1;
        while i > 0 {
            let j: usize = (self.rand() as usize) % (i + 1);
            a.swap(i,j);
            i-=1;
        }
    }
    
}
fn get_random_seed() -> Result<[u8; 32]> {
    let mut file: std::fs::File = std::fs::File::open("/dev/urandom")?;
    let mut seed: [u8; 32] = [0u8; 32];
    file.read_exact(&mut seed)?;
    Ok(seed)
}
fn convert_to_u64_array(bytes: [u8; 32]) -> [u64; 4] {
    let mut u64_array: [u64; 4] = [0u64; 4];
    for i in 0..4 {
        let start: usize = i * 8;
        let chunk: [u8; 8] = bytes[start..(start + 8)].try_into().unwrap();
        u64_array[i] = u64::from_le_bytes(chunk);
    }
    u64_array
}
fn rol64(x: u64, k: u32) -> u64 {
    (x << k) | (x >> (64 - k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xoshiro256_rand() {
        let mut rand: Xoshiro256 = Xoshiro256::from_seed([1000, 2000, 3000, 4000]);
        assert_eq!(rand.rand(), 11520000);
        assert_eq!(rand.rand(), 22855680);
        assert_eq!(rand.rand(), 1509967549440);
        assert_eq!(rand.rand(), 13474773080849308690);
        assert_eq!(rand.rand(), 6179529713436597266);
    }

    #[test]
    fn test_xoshiro256_rand_range() {
        let mut rand: Xoshiro256 = Xoshiro256::new();
        let r: u64 = rand.rand_range(1, 10);
        assert!(1 <= r && r <= 10);
    }

    #[test]
    fn test_xoshiro256_shuffle() {
        let mut rand: Xoshiro256 = Xoshiro256::from_seed([12, 22, 32, 42]);
        let mut v: Vec<i32> = vec![1, 2, 3, 4, 5];
        rand.shuffle(&mut v);
        assert_eq!(v, vec![2, 3, 4, 5, 1]);
    }

    #[test]
    fn test_xoshiro256_rand_float() {
        let mut rand: Xoshiro256 = Xoshiro256::from_seed([
            3778227730677486, 3778227799677486, 3778227799977486, 9978227730677486
        ]);
        let r: f64 = rand.rand_float();
        assert_eq!(r, 0.1797524831039718);
    }

    #[test]
    #[allow(unused_must_use)]
    fn test_xoshiro256_dieharder() {
        use std::fs::File;
        use std::io::{Write, Result};
        fn prepare_file() -> Result<()> {
            let mut rand: Xoshiro256 = Xoshiro256::new();
            let mut file: File = File::create("randomnumbers.input")?;
        
            // Write header to the file
            writeln!(file, "# seed = {} {} {} {}", rand.s[0], rand.s[1], rand.s[2], rand.s[3])?;
            writeln!(file, "type: d")?;
            writeln!(file, "count: 10000")?;
            writeln!(file, "numbit: 64")?;
        
            // Append numbers on each new line
            for _ in 0..10_000 {
                writeln!(file, "{}", rand.rand())?;
            }
            Ok(())
        }
        prepare_file();

        // Then run manually:
        // dieharder -f randomnumbers.input -a
    }
}