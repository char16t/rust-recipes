pub fn approx_equal(x: f64, y: f64, epsilon: f64) -> bool {
    return (x - y).abs() < epsilon;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_equal() {
        assert!(approx_equal(0.001, 0.002, 0.01) == true);
        assert!(approx_equal(-0.001, 0.002, 0.01) == true);
        assert!(approx_equal(-0.001, -0.002, 0.01) == true);
        assert!(approx_equal(0.000024115, 0.000023115, 0.00001) == true);
        assert!(approx_equal(0.000024115, 0.000013115, 0.00001) == false);
    }
}
