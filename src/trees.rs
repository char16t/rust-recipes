use std::collections::{HashMap, HashSet, VecDeque};

#[repr(transparent)]
pub struct AdjacencyListTree<T> {
    adjacency_list: HashMap<T, Vec<T>>
}
impl<T> AdjacencyListTree<T>
where
    T: Copy + Eq + std::hash::Hash
{
    pub fn new() -> Self {
        Self { adjacency_list: HashMap::new() }
    }
    pub fn add_child(&mut self, parent: T, child: T) {
        self.adjacency_list.entry(parent).or_insert(Vec::new()).push(child);
        self.adjacency_list.entry(child).or_insert(Vec::new()).push(parent);
    }
    pub fn iter_dfs(&self, start_node: T) -> AdjacencyListTreeDfsIterator<T> {
        AdjacencyListTreeDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs(&self, start_node: T) -> AdjacencyListTreeBfsIterator<T> {
        AdjacencyListTreeBfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn diameter(&self) -> usize {
        if let Some(start_node) = self.adjacency_list.keys().next() {
            let v: &T = start_node;
            let mut u: &T = start_node;
            let mut w: &T = start_node;

            let distances: HashMap<T, usize> = self.bfs_distances(v);
            for i in self.adjacency_list.keys() {
                if distances[i] > distances[u] {
                    u = i;
                }
            }

            let distances: HashMap<T, usize> = self.bfs_distances(u);
            for i in self.adjacency_list.keys() {
                if distances[i] > distances[w] {
                    w = i;
                }
            }

            return distances[w]
        }
        0
    }

    #[inline]
    fn bfs_distances(&self, start: &T) -> HashMap<T, usize> {
        let mut distances: HashMap<T, usize> = HashMap::new();
        let mut visited: HashSet<T> = HashSet::new();
        let mut queue: VecDeque<T> = VecDeque::new();
    
        queue.push_back(*start);
        visited.insert(*start);
        distances.insert(*start, 0);
    
        while let Some(node) = queue.pop_front() {
            for &neighbor in self.adjacency_list.get(&node).unwrap_or(&vec![]) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    distances.insert(neighbor, distances[&node] + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    
        distances
    }
}

pub struct AdjacencyListTreeDfsIterator<'a, T> {
    adjacency_list: &'a HashMap<T, Vec<T>>,
    stack: VecDeque<T>,
    visited: HashSet<T>
}
impl<'a, T> AdjacencyListTreeDfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<T, Vec<T>>, start_node: T) -> Self {
        let mut stack: VecDeque<T> = VecDeque::new();
        let mut visited: HashSet<T> = HashSet::new();
        stack.push_front(start_node);
        visited.insert(start_node);

        AdjacencyListTreeDfsIterator {
            adjacency_list,
            stack,
            visited
        }
    }
}
impl<'a, T> Iterator for AdjacencyListTreeDfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if let Some(node) = self.stack.pop_front() {
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

pub struct AdjacencyListTreeBfsIterator<'a, T> {
    adjacency_list: &'a HashMap<T, Vec<T>>,
    queue: VecDeque<T>,
    visited: HashSet<T>,
}
impl<'a, T> AdjacencyListTreeBfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<T, Vec<T>>, start_node: T) -> Self {
        let mut queue: VecDeque<T> = VecDeque::new();
        let mut visited: HashSet<T> = HashSet::new();
        queue.push_back(start_node);
        visited.insert(start_node);

        AdjacencyListTreeBfsIterator {
            adjacency_list,
            queue,
            visited,
        }
    }
}
impl<'a, T> Iterator for AdjacencyListTreeBfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if let Some(node) = self.queue.pop_front() {
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

/// Adjacency List Tree with support for switching to N ancestors
pub struct AdjacencyListTree2<T> {
    tree: AdjacencyListTree<T>,
    ancestors: AncestorGraph<T>
}
impl<T> AdjacencyListTree2<T>
where
    T: Default + Copy + Eq + std::hash::Hash
{
    pub fn new() -> Self {
        Self { tree: AdjacencyListTree::new(), ancestors: AncestorGraph::new() }
    }
    pub fn add_child(&mut self, parent: T, child: T) {
        self.tree.add_child(parent, child);
        self.ancestors.insert(child, parent);
    }
    pub fn iter_dfs(&self, start_node: T) -> AdjacencyListTreeDfsIterator<T> {
        self.tree.iter_dfs(start_node)
    }
    pub fn iter_bfs(&self, start_node: T) -> AdjacencyListTreeBfsIterator<T> {
        self.tree.iter_bfs(start_node)
    }
    pub fn diameter(&self) -> usize {
        self.tree.diameter()
    }
    pub fn ancestor(&mut self, node: T, steps: usize) -> Option<T> {
        self.ancestors.find_successor(node, steps)
    }
}

/// Successor Graph adoption for trees
pub struct AncestorGraph<T> {
    data: HashMap<T, Vec<Option<T>>>,
    is_table_filled: bool
}
impl<T> AncestorGraph<T>
where
    T: Default + Copy + Eq + std::hash::Hash
{
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            is_table_filled: false
        }
    }
    pub fn insert(&mut self, source: T, successor: T) {
        let a: &mut Vec<Option<T>> = self.data.entry(source).or_insert_with(|| {
            let mut vec: Vec<Option<T>> = Vec::with_capacity(1);
            vec.resize(1, None);
            vec
        });
        a[0] = Some(successor);
        self.is_table_filled = false;
    }
    pub fn find_successor(&mut self, node: T, steps: usize) -> Option<T> {
        if !self.is_table_filled {
            self.fill_table();
            self.is_table_filled = true;
        }

        let mut current_steps: usize = steps;
        let mut current_node: Option<T> = Some(node);
    
        while current_steps > 0 {
            let power_of_two: usize = (current_steps & !(current_steps - 1)).trailing_zeros() as usize;
            
            current_node = match current_node {
                Some(n) => self.data.get(&n).and_then(|vec| vec[power_of_two]),
                None => None
            };

            current_steps -= 1 << power_of_two;
        }

        current_node
    }
    fn fill_table(&mut self) {
        let steps: usize = self.find_max_power_of_two_not_exceeding(self.calculate_depth());
        for vec in self.data.values_mut() {
            vec.resize(steps + 1, None);
        }

        for i in 1..steps {
            let mut transformed_map: HashMap<T, Option<T>> = HashMap::new();
            for (key, values) in self.data.iter_mut() {
                if let Some(value) = values.first().cloned() {
                    transformed_map.insert(*key, value);
                }
            }
            for (&key, &opt_successor) in transformed_map.iter() {
                let mut node: Option<T> = opt_successor;
                for _ in 0..(1 << i)-1 {
                    node = match node {
                        Some(x) => transformed_map.get(&x).and_then(|inner_option| inner_option.clone()),
                        None => None 
                    };
                }
                if let Some(e) = self.data.get_mut(&key) {
                    e[i] = node;
                }
            }
        }
    }

    /// Find N that 2^N is max number not exceeding M. M is param. Returns N
    #[inline]
    fn find_max_power_of_two_not_exceeding(&self, m: usize) -> usize {
        let mut n: usize = 0;
        let mut power_of_two: usize = 1;
    
        while power_of_two * 2 <= m {
            power_of_two *= 2;
            n += 1;
        }
    
        n
    }

    #[inline]
    fn calculate_depth(&self) -> usize {
        let mut adjacency_list: HashMap<T, Vec<T>> = HashMap::new();
        for (&key, value) in self.data.iter() {
            let vec: &mut Vec<T> = adjacency_list.entry(key).or_insert(Vec::new());
            if let Some(v) = value[0] {
                vec.push(v)
            }
        }

        let mut max_depth: usize = 0;
        
        for node in adjacency_list.keys() {
            let mut visited: HashSet<T> = HashSet::new();
            if !visited.contains(node) {
                let mut stack: Vec<(T, usize)> = vec![(*node, 1)];
    
                while let Some((current_node, depth)) = stack.pop() {
                    visited.insert(current_node);
    
                    if let Some(neighbors) = adjacency_list.get(&current_node) {
                        for neighbor in neighbors {
                            if !visited.contains(neighbor) {
                                stack.push((*neighbor, depth + 1));
                            }
                        }
                    }
    
                    max_depth = max_depth.max(depth);
                }
            }
        }
        max_depth
    }
}

pub struct Node<T> {
    value: T,
    children: Vec<Node<T>>
}

impl<T> Node<T> {
    pub fn new(value: T) -> Self {
        Self { value, children: Vec::new() }
    }
    pub fn add_child(&mut self, child: Node<T>) {
        self.children.push(child);
    }
    pub fn dfs_iter(&self) -> DfsIterator<T> {
        DfsIterator::new(self)
    }
    pub fn bfs_iter(&self) -> BfsIterator<T> {
        BfsIterator::new(self)
    }
}

pub struct DfsIterator<'a, T> {
    stack: Vec<&'a Node<T>>,
}

impl<'a, T> DfsIterator<'a, T> {
    fn new(node: &'a Node<T>) -> Self {
        let mut stack: Vec<&Node<T>> = Vec::new();
        stack.push(node);
        Self { stack }
    }
}

impl<'a, T> Iterator for DfsIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            for child in node.children.iter().rev() {
                self.stack.push(child);
            }
            return Some(&node.value);
        }
        None
    }
}

pub struct BfsIterator<'a, T> {
    queue: VecDeque<&'a Node<T>>,
}

impl<'a, T> BfsIterator<'a, T> {
    fn new(node: &'a Node<T>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(node);
        BfsIterator { queue }
    }
}

impl<'a, T> Iterator for BfsIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.queue.pop_front() {
            for child in &node.children {
                self.queue.push_back(child);
            }
            Some(&node.value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_list_tree_dfs() {
        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.add_child(0, 1); tree.add_child(0, 2); tree.add_child(0, 3);
        tree.add_child(1, 11); tree.add_child(1, 12); tree.add_child(1, 13);
        tree.add_child(2, 21); tree.add_child(2, 22); tree.add_child(2, 23);
        tree.add_child(3, 31); tree.add_child(3, 32); tree.add_child(3, 33);
        tree.add_child(22, 221); tree.add_child(22, 222); tree.add_child(22, 223);

        let actual: Vec<i32> = tree.iter_dfs(0).collect();
        let expected: Vec<i32> = vec![0, 3, 33, 32, 31, 2, 23, 22, 223, 222, 221, 21, 1, 13, 12, 11];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_adjacency_list_tree_bfs() {
        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.add_child(0, 1); tree.add_child(0, 2); tree.add_child(0, 3);
        tree.add_child(1, 11); tree.add_child(1, 12); tree.add_child(1, 13);
        tree.add_child(2, 21); tree.add_child(2, 22); tree.add_child(2, 23);
        tree.add_child(3, 31); tree.add_child(3, 32); tree.add_child(3, 33);
        tree.add_child(22, 221); tree.add_child(22, 222); tree.add_child(22, 223);

        let actual: Vec<i32> = tree.iter_bfs(0).collect();
        let expected: Vec<i32> = vec![0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33, 221, 222, 223];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_adjacency_list_tree_diameter() {
        let tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 0);

        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.add_child(0, 1);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 1);

        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 2);

        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(0, 3);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 2);

        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(2, 4);
        tree.add_child(4, 5);
        tree.add_child(0, 3);
        tree.add_child(3, 6);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 5);
    }

    #[test]
    fn test_adjacency_list_tree_2_dfs() {
        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1); tree.add_child(0, 2); tree.add_child(0, 3);
        tree.add_child(1, 11); tree.add_child(1, 12); tree.add_child(1, 13);
        tree.add_child(2, 21); tree.add_child(2, 22); tree.add_child(2, 23);
        tree.add_child(3, 31); tree.add_child(3, 32); tree.add_child(3, 33);
        tree.add_child(22, 221); tree.add_child(22, 222); tree.add_child(22, 223);

        let actual: Vec<i32> = tree.iter_dfs(0).collect();
        let expected: Vec<i32> = vec![0, 3, 33, 32, 31, 2, 23, 22, 223, 222, 221, 21, 1, 13, 12, 11];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_adjacency_list_tree_2_bfs() {
        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1); tree.add_child(0, 2); tree.add_child(0, 3);
        tree.add_child(1, 11); tree.add_child(1, 12); tree.add_child(1, 13);
        tree.add_child(2, 21); tree.add_child(2, 22); tree.add_child(2, 23);
        tree.add_child(3, 31); tree.add_child(3, 32); tree.add_child(3, 33);
        tree.add_child(22, 221); tree.add_child(22, 222); tree.add_child(22, 223);

        let actual: Vec<i32> = tree.iter_bfs(0).collect();
        let expected: Vec<i32> = vec![0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33, 221, 222, 223];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_adjacency_list_tree_2_diameter() {
        let tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 0);

        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 1);

        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 2);

        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(0, 3);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 2);

        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(2, 4);
        tree.add_child(4, 5);
        tree.add_child(0, 3);
        tree.add_child(3, 6);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 5);
    }

    #[test]
    fn test_adjacency_list_tree_2_ancestors() {
        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();

        tree.add_child(0, 1); tree.add_child(0, 2); tree.add_child(0, 3);
        tree.add_child(1, 11); tree.add_child(1, 12); tree.add_child(1, 13);
        tree.add_child(2, 21); tree.add_child(2, 22); tree.add_child(2, 23);
        tree.add_child(3, 31); tree.add_child(3, 32); tree.add_child(3, 33);
        tree.add_child(22, 221); tree.add_child(22, 222); tree.add_child(22, 223);
        
        assert_eq!(tree.ancestor(0, 0), Some(0));
        assert_eq!(tree.ancestor(0, 1), None);

        assert_eq!(tree.ancestor(223, 0), Some(223));
        assert_eq!(tree.ancestor(223, 1), Some(22));
        assert_eq!(tree.ancestor(223, 2), Some(2));
        assert_eq!(tree.ancestor(223, 3), Some(0));
        assert_eq!(tree.ancestor(223, 4), None);
        assert_eq!(tree.ancestor(223, 5), None);

        assert_eq!(tree.ancestors.data[&1], vec![Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&2], vec![Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&3], vec![Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&11], vec![Some(1), Some(0), None]);
        assert_eq!(tree.ancestors.data[&12], vec![Some(1), Some(0), None]);
        assert_eq!(tree.ancestors.data[&13], vec![Some(1), Some(0), None]);
        assert_eq!(tree.ancestors.data[&21], vec![Some(2), Some(0), None]);
        assert_eq!(tree.ancestors.data[&22], vec![Some(2), Some(0), None]);
        assert_eq!(tree.ancestors.data[&23], vec![Some(2), Some(0), None]);
        assert_eq!(tree.ancestors.data[&31], vec![Some(3), Some(0), None]);
        assert_eq!(tree.ancestors.data[&32], vec![Some(3), Some(0), None]);
        assert_eq!(tree.ancestors.data[&33], vec![Some(3), Some(0), None]);
        assert_eq!(tree.ancestors.data[&221], vec![Some(22), Some(2), None]);
        assert_eq!(tree.ancestors.data[&222], vec![Some(22), Some(2), None]);
        assert_eq!(tree.ancestors.data[&223], vec![Some(22), Some(2), None]);
    }

    #[test]
    fn test_node_dfs() {
        let mut root: Node<i32> = Node::new(1);
        let mut child1: Node<i32> = Node::new(2);
        let mut child2: Node<i32> = Node::new(3);
        let child11: Node<i32> = Node::new(4);
        let child12: Node<i32> = Node::new(5);
        let child21: Node<i32> = Node::new(6);
    
        child1.add_child(child11);
        child1.add_child(child12);
        child2.add_child(child21);
    
        root.add_child(child1);
        root.add_child(child2);
    
        let actual: Vec<&i32> = root.dfs_iter().collect();
        let expected: Vec<&i32> = vec![&1, &2, &4, &5, &3, &6];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_node_bfs() {
        let mut root: Node<i32> = Node::new(1);
        let mut child1: Node<i32> = Node::new(2);
        let mut child2: Node<i32> = Node::new(3);
        let child11: Node<i32> = Node::new(4);
        let child12: Node<i32> = Node::new(5);
        let child21: Node<i32> = Node::new(6);
    
        child1.add_child(child11);
        child1.add_child(child12);
        child2.add_child(child21);
    
        root.add_child(child1);
        root.add_child(child2);
    
        let actual: Vec<&i32> = root.bfs_iter().collect();
        let expected: Vec<&i32> = vec![&1, &2, &3, &4, &5, &6];
        assert_eq!(actual, expected);
    }
}
