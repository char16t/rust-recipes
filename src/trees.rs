use std::collections::{HashMap, VecDeque};

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
    pub fn iter_dfs(&self, start_node: T) -> AdjacencyListTreeDfsIterator<T> {
        AdjacencyListTreeDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs(&self, start_node: T) -> AdjacencyListTreeBfsIterator<T> {
        AdjacencyListTreeBfsIterator::new(&self.adjacency_list, start_node)
    }
}

pub struct AdjacencyListTreeDfsIterator<'a, T> {
    adjacency_list: &'a HashMap<T, Vec<T>>,
    stack: VecDeque<T>,
    previous: Option<T>
}
impl<'a, T> AdjacencyListTreeDfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<T, Vec<T>>, start_node: T) -> Self {
        let mut stack: VecDeque<T> = VecDeque::new();
        stack.push_front(start_node);

        AdjacencyListTreeDfsIterator {
            adjacency_list,
            stack,
            previous: None
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
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    let allowed: bool = match self.previous {
                        Some(prev) => prev != neighbor,
                        None => true
                    };
                    if allowed {
                        self.stack.push_front(neighbor);
                        self.previous = Some(node);
                    }
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
    previous: Option<T>
}
impl<'a, T> AdjacencyListTreeBfsIterator<'a, T>
where
    T: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<T, Vec<T>>, start_node: T) -> Self {
        let mut queue: VecDeque<T> = VecDeque::new();
        queue.push_back(start_node);

        AdjacencyListTreeBfsIterator {
            adjacency_list,
            queue,
            previous: None
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
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    let allowed: bool = match self.previous {
                        Some(prev) => prev != neighbor,
                        None => true
                    };
                    if allowed {
                        self.queue.push_back(neighbor);
                        self.previous = Some(node);
                    }
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
        Node {
            value,
            children: Vec::new()
        }
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

    pub fn diameter(&self) -> usize {
        let mut max_diameter: usize = 0;

        let mut stack: VecDeque<(&Node<T>, usize)> = VecDeque::new();
        stack.push_back((self, 0));

        while let Some((node, height)) = stack.pop_back() {
            if height > max_diameter {
                max_diameter = height;
            }

            for child in &node.children {
                stack.push_back((child, height + 1));
            }
        }

        max_diameter
    }
}

pub struct DfsIterator<'a, T> {
    stack: Vec<&'a Node<T>>,
}

impl<'a, T> DfsIterator<'a, T> {
    fn new(node: &'a Node<T>) -> Self {
        let mut stack: Vec<&Node<T>> = Vec::new();
        stack.push(node);
        DfsIterator { stack }
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
        tree.adjacency_list.insert(0, vec![1, 2, 3]);
        tree.adjacency_list.insert(1, vec![11, 12, 13]);
        tree.adjacency_list.insert(2, vec![21, 22, 23]);
        tree.adjacency_list.insert(3, vec![31, 32, 33]);
        tree.adjacency_list.insert(22, vec![221, 222, 223]);

        let actual: Vec<i32> = tree.iter_dfs(0).collect();
        let expected: Vec<i32> = vec![0, 3, 33, 32, 31, 2, 23, 22, 223, 222, 221, 21, 1, 13, 12, 11];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_adjacency_list_tree_bfs() {
        let mut tree: AdjacencyListTree<i32> = AdjacencyListTree::new();
        tree.adjacency_list.insert(0, vec![1, 2, 3]);
        tree.adjacency_list.insert(1, vec![11, 12, 13]);
        tree.adjacency_list.insert(2, vec![21, 22, 23]);
        tree.adjacency_list.insert(3, vec![31, 32, 33]);
        tree.adjacency_list.insert(22, vec![221, 222, 223]);

        let actual: Vec<i32> = tree.iter_bfs(0).collect();
        let expected: Vec<i32> = vec![0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33, 221, 222, 223];
        assert_eq!(actual, expected);
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
        println!("{:?}", actual);
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

    #[test]
    fn test_node_diameter() {
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
        

        let diameter: usize = root.diameter();
        assert_eq!(diameter, 2);
    }
}
