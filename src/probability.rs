use crate::random;
use std::collections::HashMap;

pub struct DiscreteRandomVariable {
    rng: random::Xoshiro256,
    distribution: HashMap<i64, f64>
}
impl DiscreteRandomVariable {
    pub fn new() -> Self {
        DiscreteRandomVariable { distribution: HashMap::new(), rng: random::Xoshiro256::new() }
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
    pub fn rand(&mut self) -> i64 {
        let rand_num: f64 = self.rng.rand_float();
    
        let mut cumulative_prob: f64 = 0.0;
        for (value, &prob) in &self.distribution {
            cumulative_prob += prob;
            if cumulative_prob >= rand_num {
                return *value;
            }
        }
    
        // Return last key if for-loop ended without returning
        *self.distribution.keys().last().unwrap()
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

    #[test]
    fn test_discrete_random_variable_rand() {
        let mut random_variable: DiscreteRandomVariable = DiscreteRandomVariable::new();       
        random_variable.add(1, 0.8);
        random_variable.add(2, 0.1);
        random_variable.add(3, 0.1);

        let n: i32 = 10000;
        let mut count_map: HashMap<i64, usize> = random_variable.distribution.keys().map(|&k| (k, 0)).collect();
        for _ in 0..n {
            let random_value: i64 = random_variable.rand();
            *count_map.entry(random_value).or_insert(0) += 1;
        }
    
        let mut mean_absolute_percentage_error_sum: f64 = 0.0;

        for (value, &count) in &count_map {
            let expected_count: usize = (n as f64 * random_variable.distribution[&value]).round() as usize;
            // println!("Value: {}, Expected Count: {}, Actual Count: {}", value, expected_count, count);

            let count_difference: usize = expected_count.abs_diff(count);
            mean_absolute_percentage_error_sum += count_difference as f64 / count as f64;
        }

        let mean_absolute_percentage_error: f64 = mean_absolute_percentage_error_sum as f64 / random_variable.distribution.len() as f64;
        // println!("MAPE is {}", mean_absolute_percentage_error);

        assert!(mean_absolute_percentage_error < 0.04, "MAPE (Mean Absolute Percentage Error): {} should be < 4%", mean_absolute_percentage_error);

    }
}
