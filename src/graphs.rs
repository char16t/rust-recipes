use std::{cmp, collections::{HashMap, HashSet, VecDeque}};

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

    // Kahn’s algorithm
    pub fn topological_sort(&self) -> Option<Vec<T>> {
        let mut in_degree: HashMap<T, usize> = HashMap::new();
        let mut result: Vec<T> = Vec::new();
        let mut queue: VecDeque<T> = VecDeque::new();

        // Initialize in-degree for each node
        for neighbors in self.adjacency_list.values() {
            for neighbor in neighbors {
                *in_degree.entry(neighbor.clone()).or_insert(0) += 1;
            }
        }

        // Add nodes with in-degree 0 to the queue
        for node in self.adjacency_list.keys() {
            if !in_degree.contains_key(node) {
                queue.push_back(node.clone());
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for neighbor in neighbors {
                    *in_degree.get_mut(neighbor).unwrap() -= 1;
                    if *in_degree.get(neighbor).unwrap() == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        if result.len() == self.adjacency_list.len() {
            Some(result)
        } else {
            None // Graph contains a cycle
        }
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

    // Kahn’s algorithm
    pub fn topological_sort(&self) -> Option<Vec<T>> {
        let mut in_degree: HashMap<T, usize> = HashMap::new();
        let mut result: Vec<T> = Vec::new();
        let mut queue: VecDeque<T> = VecDeque::new();

        // Initialize in-degree for each node
        for neighbors in self.adjacency_list.values() {
            for (neighbor, _) in neighbors {
                *in_degree.entry(neighbor.clone()).or_insert(0) += 1;
            }
        }

        // Add nodes with in-degree 0 to the queue
        for node in self.adjacency_list.keys() {
            if !in_degree.contains_key(node) {
                queue.push_back(node.clone());
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for (neighbor, _) in neighbors {
                    *in_degree.get_mut(neighbor).unwrap() -= 1;
                    if *in_degree.get(neighbor).unwrap() == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        if result.len() == self.adjacency_list.len() {
            Some(result)
        } else {
            None // Graph contains a cycle
        }
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
    W: Default + Copy + Ord + std::ops::Add<Output = W>,
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
    pub fn floyd_warshall(&self) -> Vec<Option<W>> {
        let num_vertices: usize = self.capacity;
        let xy = coordinates::create_coordinate_function_2d!(self.capacity, self.capacity);
        let mut matrix: Vec<Option<W>> = vec![None; num_vertices * num_vertices];

        for i in 0..num_vertices {
            for j in 0..num_vertices {
                let coordinate: usize = xy(i, j);
                if i == j {
                    matrix[coordinate] = Some(W::default());
                } else if self.adjacency_matrix[coordinate] != W::default() {
                    matrix[coordinate] = Some(self.adjacency_matrix[coordinate]);
                }
            }
        }

        for k in 0..num_vertices {
            for i in 0..num_vertices {
                for j in 0..num_vertices {
                    // dist[i][j] = min( dist[i][j] , dist[i][k] + dist[k][j] )

                    let left: Option<W> = matrix[xy(i, j)];
                    let right_a: Option<W> = matrix[xy(i, k)];
                    let right_b: Option<W> = matrix[xy(k, j)]; 
                    
                    let right: Option<W> = if let Some(right_a_value) = right_a {
                        if let Some(right_b_value) = right_b {
                            Some(right_a_value + right_b_value)
                        } else { None }
                    } else { None };

                    let result: Option<W> = if let Some(right_value) = right {
                        if let Some(left_value) = left {
                            Some(cmp::min(left_value, right_value))
                        } else { 
                            Some(right_value) 
                        }
                    } else {
                        if let Some(left_value) = left {
                            Some(left_value)
                        } else {
                            None
                        }
                    };
                    matrix[xy(i, j)] = result;
                }
            }
        }
        matrix
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
                let &(a, b, w): &(T, T, W) = j;
                
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
                if let Some(&destination_distance) = distances.get(&destination) {
                    if source_distance + weight < destination_distance {
                        panic!("Graph contains a negative cycle");
                    }
                }
            }
        }
        distances
    }
}

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
    fn test_adjacency_list_graph_topological_sort_directed_1() {
        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_directed();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 6);
        graph.add_edge(4, 1);
        graph.add_edge(4, 5);
        graph.add_edge(5, 2);
        graph.add_edge(5, 3);

        let sorted: Option<Vec<i32>> = graph.topological_sort();
        assert_eq!(sorted, Some(vec![4, 1, 5, 2, 3, 6]));
    }

    #[test]
    fn test_adjacency_list_graph_topological_sort_directed_2() {
        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_directed();

        //cycle
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);

        graph.add_edge(3, 2);
        graph.add_edge(3, 6);
        graph.add_edge(4, 1);
        graph.add_edge(4, 5);
        graph.add_edge(5, 2);
        graph.add_edge(5, 3);

        let sorted: Option<Vec<i32>> = graph.topological_sort();
        assert_eq!(sorted, None);
    }

    #[test]
    fn test_adjacency_list_graph_topological_sort_undirected() {
        let mut graph: AdjacencyListGraph<i32> = AdjacencyListGraph::new_undirected();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 6);
        graph.add_edge(4, 1);
        graph.add_edge(4, 5);
        graph.add_edge(5, 2);
        graph.add_edge(5, 3);

        let sorted: Option<Vec<i32>> = graph.topological_sort();
        assert_eq!(sorted, None);
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
    fn test_adjacency_list_weighted_graph_topological_sort_directed_1() {
        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_directed();
        graph.add_edge(1, 2, 0.5);
        graph.add_edge(2, 3, 0.5);
        graph.add_edge(3, 6, 0.5);
        graph.add_edge(4, 1, 0.5);
        graph.add_edge(4, 5, 0.5);
        graph.add_edge(5, 2, 0.5);
        graph.add_edge(5, 3, 0.5);

        let sorted: Option<Vec<i32>> = graph.topological_sort();
        assert_eq!(sorted, Some(vec![4, 1, 5, 2, 3, 6]));
    }

    #[test]
    fn test_adjacency_list_weighted_graph_topological_sort_directed_2() {
        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_directed();

        //cycle
        graph.add_edge(1, 2, 0.5);
        graph.add_edge(2, 3, 0.5);
        graph.add_edge(3, 1, 0.5);

        graph.add_edge(3, 2, 0.5);
        graph.add_edge(3, 6, 0.5);
        graph.add_edge(4, 1, 0.5);
        graph.add_edge(4, 5, 0.5);
        graph.add_edge(5, 2, 0.5);
        graph.add_edge(5, 3, 0.5);

        let sorted: Option<Vec<i32>> = graph.topological_sort();
        assert_eq!(sorted, None);
    }

    #[test]
    fn test_adjacency_list_weighted_graph_topological_sort_undirected() {
        let mut graph: AdjacencyListWightedGraph<i32, f64> = AdjacencyListWightedGraph::new_undirected();
        graph.add_edge(1, 2, 0.5);
        graph.add_edge(2, 3, 0.5);
        graph.add_edge(3, 6, 0.5);
        graph.add_edge(4, 1, 0.5);
        graph.add_edge(4, 5, 0.5);
        graph.add_edge(5, 2, 0.5);
        graph.add_edge(5, 3, 0.5);

        let sorted: Option<Vec<i32>> = graph.topological_sort();
        assert_eq!(sorted, None);
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
        let mut graph: AdjacencyMatrixWeighted<i64> = AdjacencyMatrixWeighted::new_undirected(4);
        graph.add_edge(0, 1, 50);
        graph.add_edge(0, 2, 70);
        graph.add_edge(1, 3, 90);
        graph.add_edge(2, 3, 10);
        
        assert_eq!(graph.weigth(0, 1), 50);
        assert_eq!(graph.weigth(1, 0), 50);

        assert_eq!(graph.weigth(0, 2), 70);
        assert_eq!(graph.weigth(2, 0), 70);

        assert_eq!(graph.weigth(1, 3), 90);
        assert_eq!(graph.weigth(3, 1), 90);

        assert_eq!(graph.weigth(2, 3), 10);
        assert_eq!(graph.weigth(3, 2), 10);

        assert_eq!(graph.weigth(3, 0), 0);
    }

    #[test]
    fn test_adjacency_matrix_weighed_directed_graph() {
        let mut graph: AdjacencyMatrixWeighted<i64> = AdjacencyMatrixWeighted::new_directed(4);
        graph.add_edge(0, 1, 50);
        graph.add_edge(0, 2, 70);
        graph.add_edge(1, 3, 90);
        graph.add_edge(2, 3, 10);
        
        assert_eq!(graph.weigth(0, 1), 50);
        assert_eq!(graph.weigth(1, 0), 0);

        assert_eq!(graph.weigth(0, 2), 70);
        assert_eq!(graph.weigth(2, 0), 0);

        assert_eq!(graph.weigth(1, 3), 90);
        assert_eq!(graph.weigth(3, 1), 0);

        assert_eq!(graph.weigth(2, 3), 10);
        assert_eq!(graph.weigth(3, 2), 0);

        assert_eq!(graph.weigth(3, 0), 0);
    }

    #[test]
    fn test_adjacency_matrix_weighed_directed_graph_floyd_warshall() {
        let size: usize = 6;
        let mut graph: AdjacencyMatrixWeighted<i64> = AdjacencyMatrixWeighted::new_undirected(size);
        graph.add_edge(1, 2, 5);
        graph.add_edge(1, 4, 9);
        graph.add_edge(1, 5, 1);
        graph.add_edge(2, 3, 2);
        graph.add_edge(3, 4, 7);
        graph.add_edge(4, 5, 2);

        let xy = coordinates::create_coordinate_function_2d!(size, size);
        let matrix: Vec<Option<i64>> = graph.floyd_warshall();

        assert_eq!(matrix[xy(1, 1)], Some(0));
        assert_eq!(matrix[xy(1, 2)], Some(5));
        assert_eq!(matrix[xy(1, 3)], Some(7));
        assert_eq!(matrix[xy(1, 4)], Some(3));
        assert_eq!(matrix[xy(1, 5)], Some(1));

        assert_eq!(matrix[xy(2, 1)], Some(5));
        assert_eq!(matrix[xy(2, 2)], Some(0));
        assert_eq!(matrix[xy(2, 3)], Some(2));
        assert_eq!(matrix[xy(2, 4)], Some(8));
        assert_eq!(matrix[xy(2, 5)], Some(6));

        assert_eq!(matrix[xy(3, 1)], Some(7));
        assert_eq!(matrix[xy(3, 2)], Some(2));
        assert_eq!(matrix[xy(3, 3)], Some(0));
        assert_eq!(matrix[xy(3, 4)], Some(7));
        assert_eq!(matrix[xy(3, 5)], Some(8));

        assert_eq!(matrix[xy(4, 1)], Some(3));
        assert_eq!(matrix[xy(4, 2)], Some(8));
        assert_eq!(matrix[xy(4, 3)], Some(7));
        assert_eq!(matrix[xy(4, 4)], Some(0));
        assert_eq!(matrix[xy(4, 5)], Some(2));

        assert_eq!(matrix[xy(5, 1)], Some(1));
        assert_eq!(matrix[xy(5, 2)], Some(6));
        assert_eq!(matrix[xy(5, 3)], Some(8));
        assert_eq!(matrix[xy(5, 4)], Some(2));
        assert_eq!(matrix[xy(5, 5)], Some(0));

        // Debug :)
        // for i in 0..size {
        //     for j in 0..size {
        //         print!("{:?} ", matrix[xy(i, j)])
        //     }
        //     println!("")
        // }

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
