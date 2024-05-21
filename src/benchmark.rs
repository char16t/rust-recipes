use std::time::{Instant, Duration};

pub fn time<F, T>(f: F) -> (T, f64) where F: FnOnce() -> T {
    let start: Instant = Instant::now();
    let res: T = f();
    let end: Instant = Instant::now();

    let runtime: Duration = end.duration_since(start);
    let runtime_secs: f64 = runtime.as_secs() as f64 + f64::from(runtime.subsec_nanos()) / 1_000_000_000.0;

    (res, runtime_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time() {
        let (result, _runtime_secs) = time(|| { 32 });
        assert_eq!(result, 32);
    }
}
