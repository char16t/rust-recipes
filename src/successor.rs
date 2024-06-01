pub struct SuccessorGraphEnumerated {
    data: [[usize; 9]; 4],
}
impl SuccessorGraphEnumerated {
    pub fn new() -> Self {
        Self {
            data: [
                [2, 4, 6, 5, 1, 1, 0, 5, 2], // 1 step, A[i]
                [0, 0, 0, 0, 0, 0, 0, 0, 0], // 2 steps, A[A[i]]
                [0, 0, 0, 0, 0, 0, 0, 0, 0], // 4 steps, A[A[A[A[i]]]]
                [0, 0, 0, 0, 0, 0, 0, 0, 0], // 8 steps, A[A[A[A[A[A[A[A[i]]]]]]]
            ]
        }
    }
    pub fn find_successor(&self, idx: usize, steps: usize) -> usize {
        let mut current_steps: usize = steps;
        let mut current_idx: usize = idx;
    
        while current_steps > 0 {
            let power_of_two: u32 = (current_steps & !(current_steps - 1)).trailing_zeros();
            current_idx = self.data[power_of_two as usize][current_idx];
            current_steps -= 1 << power_of_two;
        }
    
        current_idx
    }
    pub fn fill_table(&mut self) {
    
        for i in 1..4 {
            for j in 0..9 {
                let mut idx = self.data[0][j];
                for _ in 0..(1 << i)-1 {
                    idx = self.data[0][idx];
                    //println!("idx = {}, _ = {}, [{}, {}]", idx, q, i, j);
                }
                println!("");
                self.data[i][j] = idx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_successor_graph_enumerated() {
        let mut g: SuccessorGraphEnumerated = SuccessorGraphEnumerated::new();
        g.fill_table();
        assert_eq!(g.data, [
            [2, 4, 6, 5, 1, 1, 0, 5, 2], // 1 step, A[i]
            [6, 1, 0, 1, 4, 4, 2, 1, 6], // 2 steps, A[A[i]]
            [2, 1, 6, 1, 4, 4, 0, 1, 2], // 4 steps, A[A[A[A[i]]]]
            [6, 1, 0, 1, 4, 4, 2, 1, 6], // 8 steps, A[A[A[A[A[A[A[A[i]]]]]]]
        ]);
        assert_eq!(g.data, [
            [2, 4, 6, 5, 1, 1, 0, 5, 2], // 1 step, A[i]
            [6, 1, 0, 1, 4, 4, 2, 1, 6], // 2 steps, A[A[i]]
            [2, 1, 6, 1, 4, 4, 0, 1, 2], // 4 steps, A[A[A[A[i]]]]
            [6, 1, 0, 1, 4, 4, 2, 1, 6], // 8 steps, A[A[A[A[A[A[A[A[i]]]]]]]
        ]);

        assert_eq!(g.find_successor(0, 0), 0);
        assert_eq!(g.find_successor(0, 1), 2);
        assert_eq!(g.find_successor(0, 2), 6);
        assert_eq!(g.find_successor(0, 4), 2);
        assert_eq!(g.find_successor(0, 8), 6);
        assert_eq!(g.find_successor(3, 11), 4);
    }
}
