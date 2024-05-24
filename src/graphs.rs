use std::collections::HashMap;

pub struct AdjacencyListGraph<T> {
    adjacency_list: HashMap<T, Vec<T>>,
    is_directed: bool
}

impl<T> AdjacencyListGraph<T>
where
    T: Copy + Eq + std::hash::Hash
{
    pub fn new_undirected() -> Self {
        Self { adjacency_list: HashMap::new(), is_directed: false }
    }
    pub fn new_directed() -> Self {
        Self { adjacency_list: HashMap::new(), is_directed: true }
    }
    pub fn add_edge(&mut self, a: T, b: T) {
        self.adjacency_list.entry(a).or_insert(Vec::new()).push(b);
        if !self.is_directed {
            self.adjacency_list.entry(b).or_insert(Vec::new()).push(a);
        }
    }
    pub fn neighbors(&self, v: T) -> &[T] {
        match self.adjacency_list.get(&v) {
            Some(vec) => vec,
            None => &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_list_undirected_graph() {
        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_undirected();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);

        let neighbors: &[i32] = graph.neighbors(0);
        assert_eq!(neighbors[0], 1);
        assert_eq!(neighbors[1], 2);

        let neighbors: &[i32] = graph.neighbors(1);
        assert_eq!(neighbors[0], 0);
        assert_eq!(neighbors[1], 3);

        let neighbors: &[i32] = graph.neighbors(2);
        assert_eq!(neighbors[0], 0);
        assert_eq!(neighbors[1], 3);

        let neighbors: &[i32] = graph.neighbors(3);
        assert_eq!(neighbors[0], 1);
        assert_eq!(neighbors[1], 2);

        let neighbors: &[i32] = graph.neighbors(4);
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_adjacency_list_directed_graph() {
        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_directed();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);

        let neighbors: &[i32] = graph.neighbors(0);
        assert_eq!(neighbors[0], 1);
        assert_eq!(neighbors[1], 2);

        let neighbors: &[i32] = graph.neighbors(1);
        assert_eq!(neighbors[0], 3);

        let neighbors: &[i32] = graph.neighbors(2);
        assert_eq!(neighbors[0], 3);

        let neighbors: &[i32] = graph.neighbors(3);
        assert_eq!(neighbors.len(), 0);

        let neighbors: &[i32] = graph.neighbors(4);
        assert_eq!(neighbors.len(), 0);
    }
}
