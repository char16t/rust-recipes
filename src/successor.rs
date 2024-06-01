use crate::coordinates;
use std::collections::HashMap;

pub struct SuccessorGraphEnumerated {
    data: Vec<usize>,
    size: usize,
    steps: usize
}
impl SuccessorGraphEnumerated {
    pub fn new(size: usize, steps: usize) -> Self {
        Self {
            size,
            steps: steps + 1,
            data: vec![0; (steps + 1) * size]
        }
    }
    pub fn insert(&mut self, idx: usize, successor: usize) {
        let xy = coordinates::create_coordinate_function_2d!(self.steps, self.size);
        self.data[xy(0, idx)] = successor;
    }
    pub fn find_successor(&self, idx: usize, steps: usize) -> usize {
        let xy = coordinates::create_coordinate_function_2d!(self.steps, self.size);

        let mut current_steps: usize = steps;
        let mut current_idx: usize = idx;
    
        while current_steps > 0 {
            let power_of_two: u32 = (current_steps & !(current_steps - 1)).trailing_zeros();
            current_idx = self.data[xy(power_of_two as usize, current_idx)];
            current_steps -= 1 << power_of_two;
        }
    
        current_idx
    }
    pub fn fill_table(&mut self) {
        let xy = coordinates::create_coordinate_function_2d!(self.steps, self.size);
        for i in 1..self.steps {
            for j in 0..9 {
                let mut idx = self.data[xy(0, j)];
                for _ in 0..(1 << i)-1 {
                    idx = self.data[xy(0, idx)];
                }
                self.data[xy(i, j)] = idx;
            }
        }
    }
}

pub struct SuccessorGraph<T> {
    data: HashMap<T, Vec<T>>,
    steps: usize
}
impl<T> SuccessorGraph<T>
where
    T: Default + Copy + Eq + std::hash::Hash
{
    pub fn new(steps: usize) -> Self {
        Self {
            steps: steps + 1,
            data: HashMap::new()
        }
    }
    pub fn insert(&mut self, source: T, successor: T) {
        let a: &mut Vec<T> = self.data.entry(source).or_insert_with(|| {
            let mut vec: Vec<T> = Vec::with_capacity(self.steps);
            vec.resize(self.steps, T::default());
            vec
        });
        a[0] = successor;
    }
    pub fn find_successor(&self, node: T, steps: usize) -> T {

        let mut current_steps: usize = steps;
        let mut current_node: T = node;
    
        while current_steps > 0 {
            let power_of_two: usize = (current_steps & !(current_steps - 1)).trailing_zeros() as usize;
            current_node = self.data[&current_node][power_of_two];
            current_steps -= 1 << power_of_two;
        }

        current_node
    }
    pub fn fill_table(&mut self) {
        for i in 1..self.steps {
            let mut transformed_map: HashMap<T, T> = HashMap::new();
            for (key, values) in self.data.iter_mut() {
                if let Some(value) = values.first().cloned() {
                    transformed_map.insert(*key, value);
                }
            }
            for (&key, &successor) in transformed_map.iter() {
                let mut node: T = successor;
                for _ in 0..(1 << i)-1 {
                    node = transformed_map[&node];
                }

                if let Some(e) = self.data.get_mut(&key) {
                    e[i] = node;
                }

            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_successor_graph_enumerated() {
        let size: usize = 9;
        let steps: usize = 3; // 2^3 = 8
        let mut g: SuccessorGraphEnumerated = SuccessorGraphEnumerated::new(size, steps);
        
        g.insert(0, 2);
        g.insert(1, 4);
        g.insert(2, 6);
        g.insert(3, 5);
        g.insert(4, 1);
        g.insert(5, 1);
        g.insert(6, 0);
        g.insert(7, 5);
        g.insert(8, 2);
        
        g.fill_table();

        assert_eq!(g.steps, 4);
        assert_eq!(g.data, vec![
            2, 4, 6, 5, 1, 1, 0, 5, 2,
            6, 1, 0, 1, 4, 4, 2, 1, 6,
            2, 1, 6, 1, 4, 4, 0, 1, 2,
            6, 1, 0, 1, 4, 4, 2, 1, 6,
        ]);

        assert_eq!(g.find_successor(0, 0), 0);
        assert_eq!(g.find_successor(0, 1), 2);
        assert_eq!(g.find_successor(0, 2), 6);
        assert_eq!(g.find_successor(0, 4), 2);
        assert_eq!(g.find_successor(0, 8), 6);
        assert_eq!(g.find_successor(3, 11), 4);
    }

    #[test]
    fn test_successor_graph() {
        let steps: usize = 3; // 2^3 = 8
        let mut g: SuccessorGraph<i32> = SuccessorGraph::new(steps);
        
        g.insert(0, 2);
        g.insert(1, 4);
        g.insert(2, 6);
        g.insert(3, 5);
        g.insert(4, 1);
        g.insert(5, 1);
        g.insert(6, 0);
        g.insert(7, 5);
        g.insert(8, 2);
        
        g.fill_table();

        assert_eq!(g.steps, 4);
        assert_eq!(g.data[&0], vec![2, 6, 2, 6]);
        assert_eq!(g.data[&1], vec![4, 1, 1, 1]);
        assert_eq!(g.data[&2], vec![6, 0, 6, 0]);
        assert_eq!(g.data[&3], vec![5, 1, 1, 1]);
        assert_eq!(g.data[&4], vec![1, 4, 4, 4]);
        assert_eq!(g.data[&5], vec![1, 4, 4, 4]);
        assert_eq!(g.data[&6], vec![0, 2, 0, 2]);
        assert_eq!(g.data[&7], vec![5, 1, 1, 1]);
        assert_eq!(g.data[&8], vec![2, 6, 2, 6]);

        assert_eq!(g.find_successor(0, 0), 0);
        assert_eq!(g.find_successor(0, 1), 2);
        assert_eq!(g.find_successor(0, 2), 6);
        assert_eq!(g.find_successor(0, 4), 2);
        assert_eq!(g.find_successor(0, 8), 6);
        assert_eq!(g.find_successor(3, 11), 4);
    }
}
