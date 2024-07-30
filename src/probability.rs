use std::collections::HashMap;

pub struct DiscreteRandomVariable {
    distribution: HashMap<i64, f64>
}
impl DiscreteRandomVariable {
    pub fn new() -> Self {
        DiscreteRandomVariable { distribution: HashMap::new() }
    }
    pub fn add(&mut self, value: i64, probability: f64) {
        self.distribution.insert(value, probability);
    }
    pub fn expected_value(&self) -> f64 {
        let mut expected_value: f64 = 0.0;
        for (&value, &probability) in &self.distribution {
            expected_value += probability * (value as f64)
        }
        expected_value
    }
}

#[cfg(test)]
mod tests {
    use crate::numbers;
    use super::*;

    #[test]
    fn test_discrete_random_variable_expected_value() {
        let mut cube: DiscreteRandomVariable = DiscreteRandomVariable::new();
        cube.add(1, 1.0 / 6.0);
        cube.add(2, 1.0 / 6.0);
        cube.add(3, 1.0 / 6.0);
        cube.add(4, 1.0 / 6.0);
        cube.add(5, 1.0 / 6.0);
        cube.add(6, 1.0 / 6.0);
        
        let ev: f64 = cube.expected_value();
        assert!(numbers::approx_equal(ev, 3.5, 0.00001));        
    }
}
