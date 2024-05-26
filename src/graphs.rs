use std::collections::HashMap;

use crate::{bits::{self, LongArithmetic}, coordinates, create_coordinate_function_2d};

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

pub struct AdjacencyListWightedGraph<T, W> {
    adjacency_list: HashMap<T, Vec<(T, W)>>,
    is_directed: bool
}

impl<T, W> AdjacencyListWightedGraph<T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy
{
    pub fn new_undirected() -> Self {
        Self { adjacency_list: HashMap::new(), is_directed: false }
    }
    pub fn new_directed() -> Self {
        Self { adjacency_list: HashMap::new(), is_directed: true }
    }
    pub fn add_edge(&mut self, a: T, b: T, w: W) {
        self.adjacency_list.entry(a).or_insert(Vec::new()).push((b, w));
        if !self.is_directed {
            self.adjacency_list.entry(b).or_insert(Vec::new()).push((a, w));
        }
    }
    pub fn neighbors(&self, v: T) -> &[(T, W)] {
        match self.adjacency_list.get(&v) {
            Some(vec) => vec,
            None => &[]
        }
    }
}

pub struct AdjacencyMatrixWeighted<W> {
    adjacency_matrix: Vec<W>,
    is_directed: bool,
    capacity: usize,
}
impl<W> AdjacencyMatrixWeighted<W> 
where
    W: Default + Copy + std::fmt::Debug,
{
    pub fn new_undirected(size: usize) -> Self {
        Self {
            adjacency_matrix: vec![W::default(); size * size],
            is_directed: false,
            capacity: size,
        }
    }
    pub fn new_directed(size: usize) -> Self {
        Self {
            adjacency_matrix: vec![W::default(); size * size],
            is_directed: true,
            capacity: size
        }
    }
    pub fn add_edge(&mut self, a: usize, b: usize, w: W) {
        let xy = create_coordinate_function_2d!(self.capacity, self.capacity);
        self.adjacency_matrix[xy(a, b)] = w;
        if !self.is_directed {
            self.adjacency_matrix[xy(b, a)] = w;
        }
    }
    pub fn weigth(&self, a: usize, b: usize) -> W {
        let xy = create_coordinate_function_2d!(self.capacity, self.capacity);
        let pos: usize = xy(a, b);
        return self.adjacency_matrix[pos];
    }
}

pub struct AdjacencyMatrix {
    data: bits::LongArithmetic,
    length: usize,
    is_directed: bool
}
impl AdjacencyMatrix {
    pub fn new_undirected(size: usize) -> Self {
        Self {
            data: bits::LongArithmetic::with_capacity(size),
            length: size,
            is_directed: false
        }
    }
    pub fn new_directed(size: usize) -> Self {
        Self {
            data: bits::LongArithmetic::with_capacity(size),
            length: size,
            is_directed: true
        }
    }
    pub fn add_edge(&mut self, a: usize, b: usize) {
        let xy = create_coordinate_function_2d!(self.length, self.length);
        self.data.set_bit(xy(a, b));
        if !self.is_directed {
            self.data.set_bit(xy(b, a));
        }
    }
    pub fn check_edge(&mut self, a: usize, b: usize) -> bool {
        let xy = create_coordinate_function_2d!(self.length, self.length);
        self.data.is_bit_set(xy(a, b))
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

    #[test]
    fn test_adjacency_list_weighed_undirected_graph() {
        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_undirected();
        graph.add_edge(0, 1, 0.5);
        graph.add_edge(0, 2, 0.7);
        graph.add_edge(1, 3, 0.9);
        graph.add_edge(2, 3, 0.1);

        let neighbors: &[(i32, f64)] = graph.neighbors(0);
        assert_eq!(neighbors[0], (1, 0.5));
        assert_eq!(neighbors[1], (2, 0.7));

        let neighbors: &[(i32, f64)] = graph.neighbors(1);
        assert_eq!(neighbors[0], (0, 0.5));
        assert_eq!(neighbors[1], (3, 0.9));

        let neighbors: &[(i32, f64)] = graph.neighbors(2);
        assert_eq!(neighbors[0], (0, 0.7));
        assert_eq!(neighbors[1], (3, 0.1));

        let neighbors: &[(i32, f64)] = graph.neighbors(3);
        assert_eq!(neighbors[0], (1, 0.9));
        assert_eq!(neighbors[1], (2, 0.1));

        let neighbors: &[(i32, f64)] = graph.neighbors(4);
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_adjacency_list_weighed_directed_graph() {
        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_directed();
        graph.add_edge(0, 1, 0.5);
        graph.add_edge(0, 2, 0.7);
        graph.add_edge(1, 3, 0.9);
        graph.add_edge(2, 3, 0.1);

        let neighbors: &[(i32, f64)] = graph.neighbors(0);
        assert_eq!(neighbors[0], (1, 0.5));
        assert_eq!(neighbors[1], (2, 0.7));

        let neighbors: &[(i32, f64)] = graph.neighbors(1);
        assert_eq!(neighbors[0], (3, 0.9));

        let neighbors: &[(i32, f64)] = graph.neighbors(2);
        assert_eq!(neighbors[0], (3, 0.1));

        let neighbors: &[(i32, f64)] = graph.neighbors(3);
        assert_eq!(neighbors.len(), 0);

        let neighbors: &[(i32, f64)] = graph.neighbors(4);
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_adjacency_matrix_weighed_undirected_graph() {
        let mut graph = AdjacencyMatrixWeighted::new_undirected(4);
        graph.add_edge(0, 1, 0.5);
        graph.add_edge(0, 2, 0.7);
        graph.add_edge(1, 3, 0.9);
        graph.add_edge(2, 3, 0.1);
        
        assert_eq!(graph.weigth(0, 1), 0.5);
        assert_eq!(graph.weigth(1, 0), 0.5);

        assert_eq!(graph.weigth(0, 2), 0.7);
        assert_eq!(graph.weigth(2, 0), 0.7);

        assert_eq!(graph.weigth(1, 3), 0.9);
        assert_eq!(graph.weigth(3, 1), 0.9);

        assert_eq!(graph.weigth(2, 3), 0.1);
        assert_eq!(graph.weigth(3, 2), 0.1);

        assert_eq!(graph.weigth(3, 0), 0.0);
    }


    #[test]
    fn test_adjacency_matrix_weighed_directed_graph() {
        let mut graph = AdjacencyMatrixWeighted::new_directed(4);
        graph.add_edge(0, 1, 0.5);
        graph.add_edge(0, 2, 0.7);
        graph.add_edge(1, 3, 0.9);
        graph.add_edge(2, 3, 0.1);
        
        assert_eq!(graph.weigth(0, 1), 0.5);
        assert_eq!(graph.weigth(1, 0), 0.0);

        assert_eq!(graph.weigth(0, 2), 0.7);
        assert_eq!(graph.weigth(2, 0), 0.0);

        assert_eq!(graph.weigth(1, 3), 0.9);
        assert_eq!(graph.weigth(3, 1), 0.0);

        assert_eq!(graph.weigth(2, 3), 0.1);
        assert_eq!(graph.weigth(3, 2), 0.0);

        assert_eq!(graph.weigth(3, 0), 0.0);
    }

    #[test]
    fn test_adjacency_matrix_undirected_graph() {
        let mut g: AdjacencyMatrix = AdjacencyMatrix::new_undirected(512);
        g.add_edge(256, 128);
        g.add_edge(128, 64);
        g.add_edge(64, 32);
        g.add_edge(32, 16);
        g.add_edge(16, 32);

        assert_eq!(g.check_edge(256, 128), true);
        assert_eq!(g.check_edge(128, 256), true);

        assert_eq!(g.check_edge(128, 64), true);
        assert_eq!(g.check_edge(64, 128), true);

        assert_eq!(g.check_edge(64, 32), true);
        assert_eq!(g.check_edge(32, 64), true);

        assert_eq!(g.check_edge(32, 16), true);
        assert_eq!(g.check_edge(16, 32), true);
    }

    #[test]
    fn test_adjacency_matrix_directed_graph() {
        let mut g: AdjacencyMatrix = AdjacencyMatrix::new_directed(512);
        g.add_edge(256, 128);
        g.add_edge(128, 64);
        g.add_edge(64, 32);
        g.add_edge(32, 16);
        g.add_edge(16, 32);

        assert_eq!(g.check_edge(256, 128), true);
        assert_eq!(g.check_edge(128, 256), false);

        assert_eq!(g.check_edge(128, 64), true);
        assert_eq!(g.check_edge(64, 128), false);

        assert_eq!(g.check_edge(64, 32), true);
        assert_eq!(g.check_edge(32, 64), false);

        assert_eq!(g.check_edge(32, 16), true);
        assert_eq!(g.check_edge(16, 32), true);
    }
}
