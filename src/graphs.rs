use std::collections::{HashMap, HashSet, VecDeque};

use crate::{bits, coordinates};

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
    pub fn iter_dfs(&self, start_node: T) -> AdjacencyListGraphDfsIterator<T> {
        AdjacencyListGraphDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs(&self, start_node: T) -> AdjacencyListGraphBfsIterator<T> {
        AdjacencyListGraphBfsIterator::new(&self.adjacency_list, start_node)
    }
}

pub struct AdjacencyListGraphDfsIterator<'a, T> {
    adjacency_list: &'a HashMap<T, Vec<T>>,
    visited: HashSet<T>,
    stack: VecDeque<T>,
}

impl<'a, T> AdjacencyListGraphDfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<T, Vec<T>>, start_node: T) -> Self {
        let mut stack: VecDeque<T> = VecDeque::new();
        stack.push_front(start_node);
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListGraphDfsIterator {
            adjacency_list,
            visited,
            stack,
        }
    }
}

impl<'a, T> Iterator for AdjacencyListGraphDfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while let Some(node) = self.stack.pop_front() {
            for &neighbor in &self.adjacency_list[&node] {
                if !self.visited.contains(&neighbor) {
                    self.stack.push_front(neighbor);
                    self.visited.insert(neighbor);
                }
            }
            return Some(node);
        }
        None
    }
}

pub struct AdjacencyListGraphBfsIterator<'a, T> {
    adjacency_list: &'a HashMap<T, Vec<T>>,
    visited: HashSet<T>,
    queue: VecDeque<T>,
}

impl<'a, T> AdjacencyListGraphBfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<T, Vec<T>>, start_node: T) -> Self {
        let mut queue: VecDeque<T> = VecDeque::new();
        queue.push_back(start_node);
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListGraphBfsIterator {
            adjacency_list,
            visited,
            queue,
        }
    }
}

impl<'a, T> Iterator for AdjacencyListGraphBfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while let Some(node) = self.queue.pop_front() {
            for &neighbor in &self.adjacency_list[&node] {
                if !self.visited.contains(&neighbor) {
                    self.queue.push_back(neighbor);
                    self.visited.insert(neighbor);
                }
            }
            return Some(node);
        }
        None
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
        let xy = coordinates::create_coordinate_function_2d!(self.capacity, self.capacity);
        self.adjacency_matrix[xy(a, b)] = w;
        if !self.is_directed {
            self.adjacency_matrix[xy(b, a)] = w;
        }
    }
    pub fn weigth(&self, a: usize, b: usize) -> W {
        let xy = coordinates::create_coordinate_function_2d!(self.capacity, self.capacity);
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
        let xy = coordinates::create_coordinate_function_2d!(self.length, self.length);
        self.data.set_bit(xy(a, b));
        if !self.is_directed {
            self.data.set_bit(xy(b, a));
        }
    }
    pub fn check_edge(&mut self, a: usize, b: usize) -> bool {
        let xy = coordinates::create_coordinate_function_2d!(self.length, self.length);
        self.data.is_bit_set(xy(a, b))
    }
}

pub struct EdgeListGraph<T> {
    pub edges: Vec<(T, T)>
}

impl<T> EdgeListGraph<T>
where
    T: Copy + Eq + std::hash::Hash
{
    pub fn new_directed() -> Self {
        Self { edges: Vec::new() }
    }
    pub fn add_edge(&mut self, a: T, b: T) {
        self.edges.push((a, b));
    }
}

pub struct EdgeListWegihtedGraph<T, W> {
    pub edges: Vec<(T, T, W)>
}

impl<T, W> EdgeListWegihtedGraph<T, W>
where
    T: Copy + Eq + std::hash::Hash
{
    pub fn new_directed() -> Self {
        Self { edges: Vec::new() }
    }
    pub fn add_edge(&mut self, a: T, b: T, w: W) {
        self.edges.push((a, b, w));
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

    #[test]
    fn test_edge_list_directed_graph() {
        let mut g: EdgeListGraph<i32> = EdgeListGraph::new_directed();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        assert_eq!(g.edges[0], (1, 2));
        assert_eq!(g.edges[1], (2, 3));
        assert_eq!(g.edges[2], (3, 4));
    }

    #[test]
    fn test_edge_list_weighted_directed_graph() {
        let mut g: EdgeListWegihtedGraph<i32, f64> = EdgeListWegihtedGraph::new_directed();
        g.add_edge(1, 2, 0.7);
        g.add_edge(2, 3, 0.9);
        g.add_edge(3, 4, 0.5);
        assert_eq!(g.edges[0], (1, 2, 0.7));
        assert_eq!(g.edges[1], (2, 3, 0.9));
        assert_eq!(g.edges[2], (3, 4, 0.5));
    }

    #[test]
    fn test_adjacency_list_graph_dfs_iterator() {
        let mut g: AdjacencyListGraph<i32> = AdjacencyListGraph::new_directed();
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 0);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(2, 0);
        g.add_edge(2, 5);
        g.add_edge(3, 1);
        g.add_edge(4, 1);
        g.add_edge(5, 2);

        // for node in g.iter_dfs(0) {
        //     println!("Visited Node: {}", node);
        // }

        let expected_order: Vec<i32> = vec![0, 2, 5, 1, 4, 3];
        let actual_order: Vec<i32> = g.iter_dfs(0).collect();
        assert_eq!(actual_order, expected_order);
    }

    #[test]
    fn test_adjacency_list_graph_bfs_iterator() {
        let mut g: AdjacencyListGraph<i32> = AdjacencyListGraph::new_directed();
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 0);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(2, 0);
        g.add_edge(2, 5);
        g.add_edge(3, 1);
        g.add_edge(4, 1);
        g.add_edge(5, 2);

        // for node in g.iter_bfs(0) {
        //     println!("Visited Node: {}", node);
        // }

        let expected_order: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
        let actual_order: Vec<i32> = g.iter_bfs(0).collect();
        assert_eq!(actual_order, expected_order);
    }
}
