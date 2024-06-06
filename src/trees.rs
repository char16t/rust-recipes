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
