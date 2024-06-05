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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1() {
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
    fn test_2() {
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
}
