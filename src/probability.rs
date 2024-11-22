use crate::matrices;
use crate::random;
use std::collections::HashMap;

#[derive(Default)]
pub struct DiscreteRandomVariable {
    rng: random::Xoshiro256,
    distribution: HashMap<i64, f64>,
}

impl DiscreteRandomVariable {
    pub fn new() -> Self {
        DiscreteRandomVariable {
            distribution: HashMap::new(),
            rng: random::Xoshiro256::new(),
        }
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

pub struct MarkovChain {
    matrix: matrices::Matrix<f64>,
    current: usize,
    size: usize,
    rng: random::Xoshiro256,
}
impl MarkovChain {
    pub fn new(size: usize, current: usize) -> Self {
        Self {
            matrix: matrices::Matrix::new(size, size),
            current,
            size,
            rng: random::Xoshiro256::new(),
        }
    }
    pub fn add(&mut self, from: usize, to: usize, probability: f64) {
        self.matrix[to][from] = probability;
    }
    pub fn add_bidirectional(&mut self, from: usize, to: usize, probability: f64) {
        self.matrix[to][from] = probability;
        self.matrix[from][to] = probability;
    }
    pub fn steps(&mut self, n: usize) {
        let p: matrices::Matrix<f64> = self.matrix.pow(n);
        let mut vec: Vec<f64> = vec![0.0; self.size];
        vec[self.current] = 1.0;
        let result_matrix: matrices::Matrix<f64> = p * matrices::Matrix::from_vector(&vec);

        // Choose random value from result matrix
        let rand_num: f64 = self.rng.rand_float();
        let mut cumulative_prob: f64 = 0.0;
        for i in 0..self.size {
            let prob: f64 = result_matrix[i][0];
            cumulative_prob += prob;
            if cumulative_prob >= rand_num {
                self.current = i;
                return;
            }
        }
        // Return last key if for-loop ended without returning
        self.current = self.size - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers;

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
        let mut count_map: HashMap<i64, usize> = random_variable
            .distribution
            .keys()
            .map(|&k| (k, 0))
            .collect();
        for _ in 0..n {
            let random_value: i64 = random_variable.rand();
            *count_map.entry(random_value).or_insert(0) += 1;
        }

        let mut mean_absolute_percentage_error_sum: f64 = 0.0;

        for (value, &count) in &count_map {
            let expected_count: usize =
                (n as f64 * random_variable.distribution[&value]).round() as usize;
            // println!("Value: {}, Expected Count: {}, Actual Count: {}", value, expected_count, count);

            let count_difference: usize = expected_count.abs_diff(count);
            mean_absolute_percentage_error_sum += count_difference as f64 / count as f64;
        }

        let mean_absolute_percentage_error: f64 =
            mean_absolute_percentage_error_sum as f64 / random_variable.distribution.len() as f64;
        // println!("MAPE is {}", mean_absolute_percentage_error);

        assert!(
            mean_absolute_percentage_error < 0.04,
            "MAPE (Mean Absolute Percentage Error): {} should be < 4%",
            mean_absolute_percentage_error
        );
    }

    #[test]
    fn test_markov_chain() {
        let mut chain: MarkovChain = MarkovChain::new(5, 0);
        chain.add(0, 1, 1.0);
        chain.add(1, 0, 0.5);
        chain.add_bidirectional(1, 2, 0.5);
        chain.add_bidirectional(2, 3, 0.5);
        chain.add(3, 4, 0.5);
        chain.add(4, 3, 1.0);

        chain.steps(1);
        assert_eq!(chain.current, 1);
    }
}
