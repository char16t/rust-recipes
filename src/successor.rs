use crate::coordinates;

pub struct SuccessorGraphEnumerated {
    //data: [[usize; 9]; 4],
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
            // data: [
            //     [2, 4, 6, 5, 1, 1, 0, 5, 2], // 1 step, A[i]
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0], // 2 steps, A[A[i]]
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0], // 4 steps, A[A[A[A[i]]]]
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0], // 8 steps, A[A[A[A[A[A[A[A[i]]]]]]]
            // ]
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
        for i in 1..4 {
            for j in 0..9 {
                let mut idx = self.data[xy(0, j)];
                for _ in 0..(1 << i)-1 {
                    idx = self.data[xy(0, idx)];
                    //println!("idx = {}, _ = {}, [{}, {}]", idx, q, i, j);
                }
                println!("");
                self.data[xy(i, j)] = idx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_successor_graph_enumerated() {
        let mut g: SuccessorGraphEnumerated = SuccessorGraphEnumerated::new(9, 3);
        
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
}
