use std::{cmp, collections::{HashMap, HashSet, VecDeque}, process::ExitStatus};

use crate::{bits, coordinates, heaps};

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
    pub fn add_vertex(&mut self, a: T) {
        self.adjacency_list.entry(a).or_insert(Vec::new());
    }
    pub fn add_edge(&mut self, a: T, b: T) {
        self.adjacency_list.entry(a).or_insert(Vec::new()).push(b);
        self.adjacency_list.entry(b).or_insert(Vec::new());
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
    pub fn is_connected(&self) -> bool {
        if let Some((key, _value)) = self.adjacency_list.iter().next() {
            let visited: Vec<T> = self.iter_dfs(*key).collect();
            return visited.len() == self.adjacency_list.len();
        }
        true // empty graph is connected
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
    T: Default + Copy + Eq + std::hash::Hash,
    W: Copy + Default + std::cmp::PartialOrd + std::ops::Add<Output = W>
{
    pub fn new_undirected() -> Self {
        Self { adjacency_list: HashMap::new(), is_directed: false }
    }
    pub fn new_directed() -> Self {
        Self { adjacency_list: HashMap::new(), is_directed: true }
    }
    pub fn add_vertex(&mut self, a: T) {
        self.adjacency_list.entry(a).or_insert(Vec::new());
    }
    pub fn add_edge(&mut self, a: T, b: T, w: W) {
        self.adjacency_list.entry(a).or_insert(Vec::new()).push((b, w));
        self.adjacency_list.entry(b).or_insert(Vec::new());
        if !self.is_directed {
            self.adjacency_list.entry(b).or_insert(Vec::new()).push((a, w));
        }
    }
    pub fn is_connected(&self) -> bool {
        if let Some((key, _value)) = self.adjacency_list.iter().next() {
            let visited: Vec<(T, W)> = self.iter_dfs(*key).collect();
            return visited.len() == self.adjacency_list.len();
        }
        true // empty graph is connected
    }
    pub fn neighbors(&self, v: T) -> &[(T, W)] {
        match self.adjacency_list.get(&v) {
            Some(vec) => vec,
            None => &[]
        }
    }
    pub fn iter_dfs(&self, start_node: T) -> AdjacencyListWightedGraphDfsIterator<T, W> {
        AdjacencyListWightedGraphDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs(&self, start_node: T) -> AdjacencyListWightedGraphBfsIterator<T, W> {
        AdjacencyListWightedGraphBfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_dfs_asc(&self, start_node: T) -> AdjacencyListWightedGraphAscDfsIterator<T, W> {
        AdjacencyListWightedGraphAscDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs_asc(&self, start_node: T) -> AdjacencyListWightedGraphAscBfsIterator<T, W> {
        AdjacencyListWightedGraphAscBfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_dfs_desc(&self, start_node: T) -> AdjacencyListWightedGraphDescDfsIterator<T, W> {
        AdjacencyListWightedGraphDescDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs_desc(&self, start_node: T) -> AdjacencyListWightedGraphDescBfsIterator<T, W> {
        AdjacencyListWightedGraphDescBfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn dijkstra(&self, start_node: T) -> HashMap<T, W> {
        let mut distances: HashMap<T, W> = HashMap::new();
        distances.insert(start_node, W::default());

        let mut heap: heaps::MinBinaryHeap2<T, W> = heaps::MinBinaryHeap2::with_capacity(self.adjacency_list.len());
        heap.push(start_node, W::default());

        while let Some((node, dist)) = heap.pop() {
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for (neighbor, weight) in neighbors {
                    let new_dist: W = dist + *weight;
                    let shorter_path_found: bool = match distances.get(neighbor) {
                        Some(&dist) => new_dist < dist,
                        None => true
                    };
                    if shorter_path_found {
                        distances.insert(*neighbor, new_dist);
                        heap.push(*neighbor, new_dist);
                    }
                }
            }
        }

        distances
    }
}

pub struct AdjacencyListWightedGraphDfsIterator<'a, T, W> {
    adjacency_list: &'a HashMap<T, Vec<(T, W)>>,
    visited: HashSet<T>,
    stack: VecDeque<(T, W)>,
}

impl<'a, T, W> AdjacencyListWightedGraphDfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Default
{
    fn new(adjacency_list: &'a HashMap<T, Vec<(T, W)>>, start_node: T) -> Self {
        let mut stack: VecDeque<(T, W)> = VecDeque::new();
        stack.push_front((start_node, W::default()));
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListWightedGraphDfsIterator {
            adjacency_list,
            visited,
            stack,
        }
    }
}

impl<'a, T, W> Iterator for AdjacencyListWightedGraphDfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy
{
    type Item = (T, W);

    fn next(&mut self) -> Option<(T, W)> {
        while let Some(node) = self.stack.pop_front() {
            for &neighbor in &self.adjacency_list[&node.0] {
                if !self.visited.contains(&neighbor.0) {
                    self.stack.push_front(neighbor);
                    self.visited.insert(neighbor.0);
                }
            }
            return Some(node);
        }
        None
    }
}

pub struct AdjacencyListWightedGraphBfsIterator<'a, T, W> {
    adjacency_list: &'a HashMap<T, Vec<(T, W)>>,
    visited: HashSet<T>,
    queue: VecDeque<(T, W)>,
}

impl<'a, T, W> AdjacencyListWightedGraphBfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Default
{
    fn new(adjacency_list: &'a HashMap<T, Vec<(T, W)>>, start_node: T) -> Self {
        let mut queue: VecDeque<(T, W)> = VecDeque::new();
        queue.push_back((start_node, W::default()));
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListWightedGraphBfsIterator {
            adjacency_list,
            visited,
            queue,
        }
    }
}

impl<'a, T, W> Iterator for AdjacencyListWightedGraphBfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy
{
    type Item = (T, W);

    fn next(&mut self) -> Option<(T, W)> {
        while let Some(node) = self.queue.pop_front() {
            for &neighbor in &self.adjacency_list[&node.0] {
                if !self.visited.contains(&neighbor.0) {
                    self.queue.push_back(neighbor);
                    self.visited.insert(neighbor.0);
                }
            }
            return Some(node);
        }
        None
    }
}

pub struct AdjacencyListWightedGraphAscDfsIterator<'a, T, W> {
    adjacency_list: &'a HashMap<T, Vec<(T, W)>>,
    visited: HashSet<T>,
    stack: VecDeque<(T, W)>,
}

impl<'a, T, W> AdjacencyListWightedGraphAscDfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Default
{
    fn new(adjacency_list: &'a HashMap<T, Vec<(T, W)>>, start_node: T) -> Self {
        let mut stack: VecDeque<(T, W)> = VecDeque::new();
        stack.push_front((start_node, W::default()));
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListWightedGraphAscDfsIterator {
            adjacency_list,
            visited,
            stack,
        }
    }
}

impl<'a, T, W> Iterator for AdjacencyListWightedGraphAscDfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy + Ord
{
    type Item = (T, W);

    fn next(&mut self) -> Option<(T, W)> {
        while let Some(node) = self.stack.pop_front() {
            let mut neighbors: Vec<(T, W)> = self.adjacency_list[&node.0].clone();
            neighbors.sort_by(|a, b| { b.1.cmp(&a.1) });
            for neighbor in neighbors {
                if !self.visited.contains(&neighbor.0) {
                    self.stack.push_front(neighbor);
                    self.visited.insert(neighbor.0);
                }
            }
            return Some(node);
        }
        None
    }
}

pub struct AdjacencyListWightedGraphAscBfsIterator<'a, T, W> {
    adjacency_list: &'a HashMap<T, Vec<(T, W)>>,
    visited: HashSet<T>,
    queue: VecDeque<(T, W)>,
}

impl<'a, T, W> AdjacencyListWightedGraphAscBfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Default
{
    fn new(adjacency_list: &'a HashMap<T, Vec<(T, W)>>, start_node: T) -> Self {
        let mut queue: VecDeque<(T, W)> = VecDeque::new();
        queue.push_back((start_node, W::default()));
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListWightedGraphAscBfsIterator {
            adjacency_list,
            visited,
            queue,
        }
    }
}

impl<'a, T, W> Iterator for AdjacencyListWightedGraphAscBfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy + Ord
{
    type Item = (T, W);

    fn next(&mut self) -> Option<(T, W)> {
        while let Some(node) = self.queue.pop_front() {
            let mut neighbors: Vec<(T, W)> = self.adjacency_list[&node.0].clone();
            neighbors.sort_by_key(|k| { k.1 });
            for neighbor in neighbors {
                if !self.visited.contains(&neighbor.0) {
                    self.queue.push_back(neighbor);
                    self.visited.insert(neighbor.0);
                }
            }
            return Some(node);
        }
        None
    }
}

pub struct AdjacencyListWightedGraphDescDfsIterator<'a, T, W> {
    adjacency_list: &'a HashMap<T, Vec<(T, W)>>,
    visited: HashSet<T>,
    stack: VecDeque<(T, W)>,
}

impl<'a, T, W> AdjacencyListWightedGraphDescDfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Default
{
    fn new(adjacency_list: &'a HashMap<T, Vec<(T, W)>>, start_node: T) -> Self {
        let mut stack: VecDeque<(T, W)> = VecDeque::new();
        stack.push_front((start_node, W::default()));
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListWightedGraphDescDfsIterator {
            adjacency_list,
            visited,
            stack,
        }
    }
}

impl<'a, T, W> Iterator for AdjacencyListWightedGraphDescDfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy + Ord
{
    type Item = (T, W);

    fn next(&mut self) -> Option<(T, W)> {
        while let Some(node) = self.stack.pop_front() {
            let mut neighbors: Vec<(T, W)> = self.adjacency_list[&node.0].clone();
            neighbors.sort_by(|a, b| a.1.cmp(&b.1));
            for neighbor in neighbors {
                if !self.visited.contains(&neighbor.0) {
                    self.stack.push_front(neighbor);
                    self.visited.insert(neighbor.0);
                }
            }
            return Some(node);
        }
        None
    }
}

pub struct AdjacencyListWightedGraphDescBfsIterator<'a, T, W> {
    adjacency_list: &'a HashMap<T, Vec<(T, W)>>,
    visited: HashSet<T>,
    queue: VecDeque<(T, W)>,
}

impl<'a, T, W> AdjacencyListWightedGraphDescBfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Default
{
    fn new(adjacency_list: &'a HashMap<T, Vec<(T, W)>>, start_node: T) -> Self {
        let mut queue: VecDeque<(T, W)> = VecDeque::new();
        queue.push_back((start_node, W::default()));
        let mut visited: HashSet<T> = HashSet::new();
        visited.insert(start_node);

        AdjacencyListWightedGraphDescBfsIterator {
            adjacency_list,
            visited,
            queue,
        }
    }
}

impl<'a, T, W> Iterator for AdjacencyListWightedGraphDescBfsIterator<'a, T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy + Ord
{
    type Item = (T, W);

    fn next(&mut self) -> Option<(T, W)> {
        while let Some(node) = self.queue.pop_front() {
            let mut neighbors: Vec<(T, W)> = self.adjacency_list[&node.0].clone();
            neighbors.sort_by(|a, b| b.1.cmp(&a.1));
            for neighbor in neighbors {
                if !self.visited.contains(&neighbor.0) {
                    self.queue.push_back(neighbor);
                    self.visited.insert(neighbor.0);
                }
            }
            return Some(node);
        }
        None
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
    pub edges: Vec<(T, T, W)>,
    is_directed: bool
}

impl<T, W> EdgeListWegihtedGraph<T, W>
where
    T: Copy + Eq + std::hash::Hash,
    W: Copy + Default + std::cmp::Ord + std::ops::Add<W, Output = W>
{
    pub fn new_undirected() -> Self {
        Self { edges: Vec::new(), is_directed: false }
    }
    pub fn new_directed() -> Self {
        Self { edges: Vec::new(), is_directed: true }
    }
    pub fn add_edge(&mut self, a: T, b: T, w: W) {
        self.edges.push((a, b, w));
        if !self.is_directed {
            self.edges.push((b, a, w));
        }
    }
    pub fn bellman_ford(&self, start_node: T, n_vertices: usize) -> HashMap<T, W> {
        let mut distances: HashMap<T, W> = HashMap::new();
        distances.insert(start_node, W::default());
        for _ in 0..n_vertices-1 {
            for j in self.edges.iter() {
                let &(a, b, w) = j;
                
                // distance[b] = min(distance[b], distance[a]+w);
                if let Some(&distance_b) = distances.get(&b) {
                    if let Some(&distance_a) = distances.get(&a) {
                        distances.insert(b, cmp::min(distance_b, distance_a + w));
                    } else {
                        // distance_a = INF
                        distances.insert(b, distance_b);
                    }
                } else {
                    if let Some(&distance_a) = distances.get(&a) {
                         // distance_b = INF
                         distances.insert(b, distance_a + w);
                    }
                    // else distance_a = INF && distance_b = INF
                }
            }
        }
        for &edge in self.edges.iter() {
            let (source, destination, weight) = edge;
            if let Some(&source_distance) = distances.get(&source) {
                if let Some(&destination_distance) = distances.get(&source) {
                    if source_distance + weight < destination_distance {
                        panic!("Graph contains a negative cycle");
                    }
                }
            }
        }
        distances
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
    fn test_adjacency_list_graph_is_connected() {
        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_undirected();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);
        assert_eq!(graph.is_connected(), true);

        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_undirected();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        assert_eq!(graph.is_connected(), true);

        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_undirected();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1);
        assert_eq!(graph.is_connected(), false);

        let empty: AdjacencyListGraph<i32> = AdjacencyListGraph::new_undirected();
        assert_eq!(empty.is_connected(), true);
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
    fn test_adjacency_list_weighted_graph_is_connected() {
        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_undirected();
        graph.add_edge(0, 1, 0.1);
        graph.add_edge(0, 2, 0.1);
        graph.add_edge(1, 3, 0.1);
        graph.add_edge(2, 3, 0.1);
        assert_eq!(graph.is_connected(), true);

        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_undirected();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 0.1);
        graph.add_edge(0, 2, 0.1);
        assert_eq!(graph.is_connected(), true);

        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_undirected();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 0.1);
        assert_eq!(graph.is_connected(), false);

        let empty: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_undirected();
        assert_eq!(empty.is_connected(), true);
    }

    #[test]
    fn test_adjacency_list_weighted_graph_dijkstra() {
        let mut graph: AdjacencyListWightedGraph<char, f64> = AdjacencyListWightedGraph::new_undirected();
        graph.add_edge('A', 'B', 10.0);
        graph.add_edge('A', 'C', 5.0);
        graph.add_edge('B', 'D', 25.0);
        graph.add_edge('C', 'D', 50.0);
        let distances: HashMap<char, f64> = graph.dijkstra('A');
        assert_eq!(distances[&'A'], 0.0);
        assert_eq!(distances[&'B'], 10.0);
        assert_eq!(distances[&'C'], 5.0);
        assert_eq!(distances[&'D'], 35.0);
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
        let mut g: EdgeListWegihtedGraph<i32, i64> = EdgeListWegihtedGraph::new_directed();
        g.add_edge(1, 2, 70);
        g.add_edge(2, 3, 90);
        g.add_edge(3, 4, 50);
        assert_eq!(g.edges[0], (1, 2, 70));
        assert_eq!(g.edges[1], (2, 3, 90));
        assert_eq!(g.edges[2], (3, 4, 50));
    }

    #[test]
    fn test_edge_list_weighted_undirected_graph() {
        let mut g: EdgeListWegihtedGraph<i32, i64> = EdgeListWegihtedGraph::new_undirected();
        g.add_edge(1, 2, 70);
        g.add_edge(2, 3, 90);
        g.add_edge(3, 4, 50);
        assert_eq!(g.edges[0], (1, 2, 70));
        assert_eq!(g.edges[1], (2, 1, 70));
        assert_eq!(g.edges[2], (2, 3, 90));
        assert_eq!(g.edges[3], (3, 2, 90));
        assert_eq!(g.edges[4], (3, 4, 50));
        assert_eq!(g.edges[5], (4, 3, 50));
    }

    #[test]
    fn test_edge_list_weighted_directed_graph_bellman_ford() {
        let mut g: EdgeListWegihtedGraph<i32, i64> = EdgeListWegihtedGraph::new_directed();
        g.add_edge(1, 2, 10);
        g.add_edge(1, 3, 5);
        g.add_edge(2, 4, 25);
        g.add_edge(3, 4, 50);
        
        let start_node: i32 = 1;
        let n_vertices: usize = 4;
        let distances: HashMap<i32, i64> = g.bellman_ford(start_node, n_vertices);
        assert_eq!(distances[&1], 0);
        assert_eq!(distances[&2], 10);
        assert_eq!(distances[&3], 5);
        assert_eq!(distances[&4], 35);
    }

    #[test]
    #[should_panic]
    fn test_edge_list_weighted_directed_graph_bellman_ford_negative_cycle() {
        let mut g: EdgeListWegihtedGraph<i32, i64> = EdgeListWegihtedGraph::new_undirected();
        g.add_edge(1, 2, 3);
        g.add_edge(1, 3, 5);
        g.add_edge(2, 3, 2);
        g.add_edge(2, 4, 1);
        g.add_edge(3, 4, -7);

        // negative cycle: 2 → 3 → 4 → 2
        
        let start_node: i32 = 2;
        let n_vertices: usize = 4;
        g.bellman_ford(start_node, n_vertices);
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

    #[test]
    fn test_adjacency_list_weighted_graph_dfs_iterator() {

        let mut g: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_directed();
        g.add_edge(0, 1, 0.1);
        g.add_edge(0, 2, 0.2);
        g.add_edge(1, 0, 0.3);
        g.add_edge(1, 3, 0.4);
        g.add_edge(1, 4, 0.5);
        g.add_edge(2, 0, 0.6);
        g.add_edge(2, 5, 0.7);
        g.add_edge(3, 1, 0.8);
        g.add_edge(4, 1, 0.9);
        g.add_edge(5, 2, 0.25);

        // for node in g.iter_dfs(0) {
        //     println!("Visited Node: {}", node);
        // }

        let expected_order: Vec<(i32, f64)> = vec![(0, 0.0), (2, 0.2), (5, 0.7), (1, 0.1), (4, 0.5), (3, 0.4)];
        let actual_order: Vec<(i32, f64)> = g.iter_dfs(0).collect();
        assert_eq!(actual_order, expected_order);
    }

    #[test]
    fn ttest_adjacency_list_weighted_bfs_iterator() {
        let mut g: AdjacencyListWightedGraph<i32, i64> = AdjacencyListWightedGraph::new_directed();
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 20);
        g.add_edge(1, 0, 30);
        g.add_edge(1, 3, 40);
        g.add_edge(1, 4, 50);
        g.add_edge(2, 0, 60);
        g.add_edge(2, 5, 70);
        g.add_edge(3, 1, 80);
        g.add_edge(4, 1, 90);
        g.add_edge(5, 2, 25);

        // for node in g.iter_bfs(0) {
        //     println!("Visited Node: {}", node);
        // }

        let expected_order: Vec<(i32, i64)> = vec![(0, 0), (1, 10), (2, 20), (3, 40), (4, 50), (5, 70)];
        let actual_order: Vec<(i32, i64)> = g.iter_bfs(0).collect();
        assert_eq!(actual_order, expected_order);
    }

    #[test]
    fn test_adjacency_list_weighted_graph_asc_dfs_iterator() {
        let mut g: AdjacencyListWightedGraph<i32, i64> = AdjacencyListWightedGraph::new_directed();
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 20);
        g.add_edge(1, 0, 30);
        g.add_edge(1, 3, 40);
        g.add_edge(1, 4, 50);
        g.add_edge(2, 0, 60);
        g.add_edge(2, 5, 70);
        g.add_edge(3, 1, 80);
        g.add_edge(4, 1, 90);
        g.add_edge(5, 2, 25);

        // for node in g.iter_dfs_asc(0) {
        //     println!("Visited Node: {}", node);
        // }

        let expected_order: Vec<(i32, i64)> = vec![(0, 0), (1, 10), (3, 40), (4, 50), (2, 20), (5, 70)];
        let actual_order: Vec<(i32, i64)> = g.iter_dfs_asc(0).collect();
        assert_eq!(actual_order, expected_order);
    }

    #[test]
    fn test_adjacency_list_weighted_asc_bfs_iterator() {
        let mut g: AdjacencyListWightedGraph<i32, i64> = AdjacencyListWightedGraph::new_directed();
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 20);
        g.add_edge(1, 0, 30);
        g.add_edge(1, 3, 40);
        g.add_edge(1, 4, 50);
        g.add_edge(2, 0, 60);
        g.add_edge(2, 5, 70);
        g.add_edge(3, 1, 80);
        g.add_edge(4, 1, 90);
        g.add_edge(5, 2, 25);

        // for node in g.iter_bfs_asc(0) {
        //     println!("Visited Node: {} ({})", node.0, node.1);
        // }

        let expected_order: Vec<(i32, i64)> = vec![(0, 0), (1, 10), (2, 20), (3, 40), (4, 50), (5, 70)];
        let actual_order: Vec<(i32, i64)> = g.iter_bfs_asc(0).collect();
        assert_eq!(actual_order, expected_order);
    }

    #[test]
    fn test_adjacency_list_weighted_graph_desc_dfs_iterator() {
        let mut g: AdjacencyListWightedGraph<i32, i64> = AdjacencyListWightedGraph::new_directed();
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 20);
        g.add_edge(1, 0, 30);
        g.add_edge(1, 3, 40);
        g.add_edge(1, 4, 50);
        g.add_edge(2, 0, 60);
        g.add_edge(2, 5, 70);
        g.add_edge(3, 1, 80);
        g.add_edge(4, 1, 90);
        g.add_edge(5, 2, 25);

        // for node in g.iter_dfs_desc(0) {
        //     println!("Visited Node: {}", node);
        // }

        let expected_order: Vec<(i32, i64)> = vec![(0, 0), (2, 20), (5, 70), (1, 10), (4, 50), (3, 40)];
        let actual_order: Vec<(i32, i64)> = g.iter_dfs_desc(0).collect();
        assert_eq!(actual_order, expected_order);
    }

    #[test]
    fn test_adjacency_list_weighted_desc_bfs_iterator() {
        let mut g: AdjacencyListWightedGraph<i32, i64> = AdjacencyListWightedGraph::new_directed();
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 20);
        g.add_edge(1, 0, 30);
        g.add_edge(1, 3, 40);
        g.add_edge(1, 4, 50);
        g.add_edge(2, 0, 60);
        g.add_edge(2, 5, 70);
        g.add_edge(3, 1, 80);
        g.add_edge(4, 1, 90);
        g.add_edge(5, 2, 25);

        // for node in g.iter_bfs_desc(0) {
        //     println!("Visited Node: {} ({})", node.0, node.1);
        // }

        let expected_order: Vec<(i32, i64)> = vec![(0, 0), (2, 20), (1, 10), (5, 70), (4, 50), (3, 40)];
        let actual_order: Vec<(i32, i64)> = g.iter_bfs_desc(0).collect();
        assert_eq!(actual_order, expected_order);
    }
}
