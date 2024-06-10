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
            for &neighbor in self.adjacency_list[&node].iter().rev() {
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
    pub fn lowest_common_ancestor(&mut self, a: T, b: T) -> T {
        let mut a_depth: usize = 0;
        while let Some(_) = self.ancestor(a, a_depth) {
            a_depth += 1;
        }
        a_depth -= 1;

        let mut b_depth: usize = 0;
        while let Some(_) = self.ancestor(b, b_depth) {
            b_depth += 1;
        }
        b_depth -= 1;

        let mut node_a: Option<T> = Some(a);
        let mut node_b: Option<T> = Some(b);
        if a_depth > b_depth {
            node_a = self.ancestor(a, a_depth - b_depth);
        } else if b_depth > a_depth {
            node_b = self.ancestor(b, b_depth - a_depth);
        }

        let mut depth: usize = 1;
        while depth <= a_depth {
            if let (Some(aa), Some(bb)) = (node_a, node_b) {
                if aa == bb {
                    return aa;
                }
            }
            node_a = match node_a {
                Some(a) => self.ancestor(a, depth),
                None => None
            };
            node_b = match node_b {
                Some(b) => self.ancestor(b, depth),
                None => None
            };
            depth += 1;
        }

        node_a.unwrap()
    }
    pub fn distance(&mut self, a: T, b: T) -> usize {
        let mut a_depth: usize = 0;
        while let Some(_) = self.ancestor(a, a_depth) {
            a_depth += 1;
        }
        a_depth -= 1;

        let mut b_depth: usize = 0;
        while let Some(_) = self.ancestor(b, b_depth) {
            b_depth += 1;
        }
        b_depth -= 1;

        let mut node_a: Option<T> = Some(a);
        let mut node_b: Option<T> = Some(b);
        if a_depth > b_depth {
            node_a = self.ancestor(a, a_depth - b_depth);
        } else if b_depth > a_depth {
            node_b = self.ancestor(b, b_depth - a_depth);
        }

        let max_depth: usize = a_depth.max(b_depth);
        let mut depth: usize = 0;
        let mut lca: T = node_a.unwrap();
        loop {
            if let (Some(aa), Some(bb)) = (node_a, node_b) {
                if aa == bb {
                    lca = aa;
                    break;
                }
            }
            if depth > max_depth {
                break;
            }

            node_a = match node_a {
                Some(a) => self.ancestor(a, depth),
                None => None
            };
            node_b = match node_b {
                Some(b) => self.ancestor(b, depth),
                None => None
            };
            depth += 1;
        }
        if depth > max_depth {
            lca = if a_depth <= b_depth {
                node_a.unwrap()
            } else {
                node_b.unwrap()
            };
        }

        let mut lca_depth: usize = 0;
        while let Some(_) = self.ancestor(lca, lca_depth) {
            lca_depth += 1;
        }
        lca_depth -= 1;

        a_depth + b_depth - 2*lca_depth
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
        let steps: usize = self.find_max_power_of_two_not_exceeding(self.calculate_depth()) + 1;
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

#[repr(transparent)]
pub struct KeyValueTree<K, V> {
    adjacency_list: HashMap<K, (V, Vec<K>)>,
}
impl<K, V> KeyValueTree<K, V>
where
    K: Copy + Eq + std::hash::Hash,
    V: Default + Copy
{
    pub fn new() -> Self {
        Self { adjacency_list: HashMap::new() }
    }
    pub fn add_root(&mut self, parent: K, parent_value: V) {
        self.adjacency_list.entry(parent).or_insert((parent_value, Vec::new())).0 = parent_value;
    }
    pub fn add_child(&mut self, parent: K, child: K, child_value: V) {
        self.adjacency_list.entry(parent).or_insert((V::default(), Vec::new())).1.push(child);

        let c: &mut (V, Vec<K>) = self.adjacency_list.entry(child).or_insert((child_value, Vec::new()));
        c.1.push(parent);
        c.0 = child_value;
    }
    pub fn iter_dfs(&self, start_node: K) -> KeyValueTreeDfsIterator<K, V> {
        KeyValueTreeDfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn iter_bfs(&self, start_node: K) -> KeyValueTreeBfsIterator<K, V> {
        KeyValueTreeBfsIterator::new(&self.adjacency_list, start_node)
    }
    pub fn diameter(&self) -> usize {
        if let Some(start_node) = self.adjacency_list.keys().next() {
            let v: &K = start_node;
            let mut u: &K = start_node;
            let mut w: &K = start_node;

            let distances: HashMap<K, usize> = self.bfs_distances(v);
            for i in self.adjacency_list.keys() {
                if distances[i] > distances[u] {
                    u = i;
                }
            }

            let distances: HashMap<K, usize> = self.bfs_distances(u);
            for i in self.adjacency_list.keys() {
                if distances[i] > distances[w] {
                    w = i;
                }
            }

            return distances[w]
        }
        0
    }

    /// Return tuple of: 1) Map<Node, Index in array>, 2) Vector of pair: (subtree size, value in node)
    pub fn dfs_with_subtree_sizes(&self, start_node: K) -> (HashMap<K, usize>, Vec<(usize, V)>) {
        let mut stack: Vec<(K, Option<K>)> = vec![(start_node, None)];
        let mut visited: HashSet<K> = HashSet::new();
        let mut subtree_sizes: HashMap<K, usize> = HashMap::new();
        while let Some((node, parent)) = stack.pop() {
            if visited.contains(&node) {
                if let Some(parent_node) = parent {
                    let sub_size: usize = *subtree_sizes.get(&node).unwrap_or(&1);
                    subtree_sizes.entry(parent_node).and_modify(|x| *x += sub_size);

                }
            } else {
                visited.insert(node);
                stack.push((node, parent));
                for &neighbor in self.adjacency_list.get(&node).and_then(|x| Some(&x.1)).unwrap_or(&vec![]).iter().rev() {
                    if !visited.contains(&neighbor) {
                        stack.push((neighbor, Some(node)));
                    }
                }
                subtree_sizes.insert(node, 1);
            }
        }
    
        let mut result_key_ids: HashMap<K, usize> = HashMap::new();
        let mut result: Vec<(usize, V)> = Vec::new(); // i
        let mut stack: Vec<K> = Vec::new();
        let mut visited: HashSet<K> = HashSet::new();
        stack.push(start_node);
        visited.insert(start_node);
        let mut index: usize = 0;
        while let Some(node) = stack.pop() {
            if let Some((value, neighbors)) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors.iter().rev() {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
                result_key_ids.insert(node, index);
                result.push((subtree_sizes[&node], *value));
                index += 1;
            }
        }
        (result_key_ids, result)
    }

    #[inline]
    fn bfs_distances(&self, start: &K) -> HashMap<K, usize> {
        let mut distances: HashMap<K, usize> = HashMap::new();
        let mut visited: HashSet<K> = HashSet::new();
        let mut queue: VecDeque<K> = VecDeque::new();
    
        queue.push_back(*start);
        visited.insert(*start);
        distances.insert(*start, 0);
    
        while let Some(node) = queue.pop_front() {
            for &neighbor in self.adjacency_list.get(&node).and_then(|x| Some(&x.1)).unwrap_or(&vec![]) {
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

pub struct KeyValueTreeDfsIterator<'a, K, V> {
    adjacency_list: &'a HashMap<K, (V, Vec<K>)>,
    stack: VecDeque<K>,
    visited: HashSet<K>
}
impl<'a, K, V> KeyValueTreeDfsIterator<'a, K, V>
where
    K: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<K, (V, Vec<K>)>, start_node: K) -> Self {
        let mut stack: VecDeque<K> = VecDeque::new();
        let mut visited: HashSet<K> = HashSet::new();
        stack.push_front(start_node);
        visited.insert(start_node);

        KeyValueTreeDfsIterator {
            adjacency_list,
            stack,
            visited
        }
    }
}
impl<'a, K, V> Iterator for KeyValueTreeDfsIterator<'a, K, V>
where
    K: Copy + Eq + std::hash::Hash,
    V: Copy
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if let Some(node) = self.stack.pop_front() {
            if let Some((value, neighbors)) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors.iter().rev() {
                    if !self.visited.contains(&neighbor) {
                        self.stack.push_front(neighbor);
                        self.visited.insert(neighbor);
                    }
                }
                return Some((node, *value));
            }
        }
        None
    }
}

pub struct KeyValueTreeBfsIterator<'a, K, V> {
    adjacency_list: &'a HashMap<K, (V, Vec<K>)>,
    queue: VecDeque<K>,
    visited: HashSet<K>,
}
impl<'a, K, V> KeyValueTreeBfsIterator<'a, K, V>
where
    K: Copy + Eq + std::hash::Hash
{
    fn new(adjacency_list: &'a HashMap<K, (V, Vec<K>)>, start_node: K) -> Self {
        let mut queue: VecDeque<K> = VecDeque::new();
        let mut visited: HashSet<K> = HashSet::new();
        queue.push_back(start_node);
        visited.insert(start_node);

        KeyValueTreeBfsIterator {
            adjacency_list,
            queue,
            visited,
        }
    }
}
impl<'a, K, V> Iterator for KeyValueTreeBfsIterator<'a, K, V>
where
    K: Copy + Eq + std::hash::Hash,
    V: Copy
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if let Some(node) = self.queue.pop_front() {
            if let Some((value, neighbors)) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    if !self.visited.contains(&neighbor) {
                        self.queue.push_back(neighbor);
                        self.visited.insert(neighbor);
                    }
                }
                return Some((node, *value));
            }
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
        let expected: Vec<i32> = vec![0, 1, 11, 12, 13, 2, 21, 22, 221, 222, 223, 23, 3, 31, 32, 33];
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
        let expected: Vec<i32> = vec![0, 1, 11, 12, 13, 2, 21, 22, 221, 222, 223, 23, 3, 31, 32, 33];
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

        assert_eq!(tree.ancestors.data[&1], vec![Some(0), None, None, None]);
        assert_eq!(tree.ancestors.data[&2], vec![Some(0), None, None, None]);
        assert_eq!(tree.ancestors.data[&3], vec![Some(0), None, None, None]);
        assert_eq!(tree.ancestors.data[&11], vec![Some(1), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&12], vec![Some(1), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&13], vec![Some(1), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&21], vec![Some(2), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&22], vec![Some(2), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&23], vec![Some(2), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&31], vec![Some(3), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&32], vec![Some(3), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&33], vec![Some(3), Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&221], vec![Some(22), Some(2), None, None]);
        assert_eq!(tree.ancestors.data[&222], vec![Some(22), Some(2), None, None]);
        assert_eq!(tree.ancestors.data[&223], vec![Some(22), Some(2), None, None]);
    }

    #[test]
    fn test_adjacency_list_tree_2_ancestors_small_tree() {
        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();

        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(1, 11);
        tree.add_child(2, 22);

        assert_eq!(tree.ancestor(11, 0), Some(11));
        assert_eq!(tree.ancestor(11, 1), Some(1));
        assert_eq!(tree.ancestor(11, 2), Some(0));
        assert_eq!(tree.ancestor(11, 3), None);
        assert_eq!(tree.ancestor(11, 4), None);
        assert_eq!(tree.ancestor(11, 5), None);

        assert_eq!(tree.ancestor(22, 0), Some(22));
        assert_eq!(tree.ancestor(22, 1), Some(2));
        assert_eq!(tree.ancestor(22, 2), Some(0));
        assert_eq!(tree.ancestor(22, 3), None);
        assert_eq!(tree.ancestor(22, 4), None);
        assert_eq!(tree.ancestor(22, 5), None);

        assert_eq!(tree.ancestors.data[&1], vec![Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&2], vec![Some(0), None, None]);
        assert_eq!(tree.ancestors.data[&11], vec![Some(1), Some(0), None]);
        assert_eq!(tree.ancestors.data[&22], vec![Some(2), Some(0), None]);
    }

    #[test]
    fn test_adjacency_list_tree_2_lowest_common_ancestor() {
        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(1, 11);
        tree.add_child(2, 22);

        let lca: i32 = tree.lowest_common_ancestor(0, 0);
        assert_eq!(lca, 0);

        let lca: i32 = tree.lowest_common_ancestor(1, 2);
        assert_eq!(lca, 0);

        let lca: i32 = tree.lowest_common_ancestor(1, 0);
        assert_eq!(lca, 0);

        let lca: i32 = tree.lowest_common_ancestor(0, 2);
        assert_eq!(lca, 0);

        let lca: i32 = tree.lowest_common_ancestor(1, 11);
        assert_eq!(lca, 1);

        let lca: i32 = tree.lowest_common_ancestor(11, 1);
        assert_eq!(lca, 1);

        let lca: i32 = tree.lowest_common_ancestor(2, 11);
        assert_eq!(lca, 0);

        let lca: i32 = tree.lowest_common_ancestor(22, 0);
        assert_eq!(lca, 0);
    }

    #[test]
    fn test_adjacency_list_tree_2_distance() {
        let mut tree: AdjacencyListTree2<i32> = AdjacencyListTree2::new();
        tree.add_child(0, 1);
        tree.add_child(0, 2);
        tree.add_child(1, 11);
        tree.add_child(2, 22);

        let dist: usize = tree.distance(0, 0);
        assert_eq!(dist, 0);

        let dist: usize = tree.distance(0, 1);
        assert_eq!(dist, 1);

        let dist: usize = tree.distance(2, 0);
        assert_eq!(dist, 1);

        let dist: usize = tree.distance(1, 2);
        assert_eq!(dist, 2);

        let dist: usize = tree.distance(2, 11);
        assert_eq!(dist, 3);

        let dist: usize = tree.distance(22, 1);
        assert_eq!(dist, 3);

        let dist: usize = tree.distance(0, 11);
        assert_eq!(dist, 2);

        let dist: usize = tree.distance(22, 0);
        assert_eq!(dist, 2);

        let dist: usize = tree.distance(11, 11);
        assert_eq!(dist, 0);

        let dist: usize = tree.distance(1, 11);
        assert_eq!(dist, 0);
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

    #[test]
    fn test_key_value_tree_bfs() {
        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_root(1, 2);
        tree.add_child(1, 2, 3);
        tree.add_child(1, 3, 5);
        tree.add_child(1, 4, 3);
        tree.add_child(1, 5, 1);
        tree.add_child(2, 6, 4);
        tree.add_child(4, 7, 4);
        tree.add_child(4, 8, 3);
        tree.add_child(4, 9, 1);

        let res: Vec<(i32, usize)> = tree.iter_bfs(1).collect();
        assert_eq!(res, vec![(1, 2), (2, 3), (3, 5), (4, 3), (5, 1), (6, 4), (7, 4), (8, 3), (9, 1)]);
    }

    #[test]
    fn test_key_value_tree_dfs() {
        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_root(1, 2);
        tree.add_child(1, 2, 3);
        tree.add_child(1, 3, 5);
        tree.add_child(1, 4, 3);
        tree.add_child(1, 5, 1);
        tree.add_child(2, 6, 4);
        tree.add_child(4, 7, 4);
        tree.add_child(4, 8, 3);
        tree.add_child(4, 9, 1);

        let res: Vec<(i32, usize)> = tree.iter_dfs(1).collect();
        assert_eq!(res, vec![(1, 2), (2, 3), (6, 4), (3, 5), (4, 3), (7, 4), (8, 3), (9, 1), (5, 1)]);
    }

    #[test]
    fn test_key_value_tree_diameter() {
        let tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 0);

        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_child(0, 1, 1);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 1);

        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_child(0, 1, 1);
        tree.add_child(0, 2, 1);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 2);

        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_child(0, 1, 1);
        tree.add_child(0, 2, 1);
        tree.add_child(0, 3, 1);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 2);

        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_child(0, 1, 1);
        tree.add_child(0, 2, 1);
        tree.add_child(2, 4, 1);
        tree.add_child(4, 5, 1);
        tree.add_child(0, 3, 1);
        tree.add_child(3, 6, 1);
        let diameter: usize = tree.diameter();
        assert_eq!(diameter, 5);
    }

    #[test]
    fn test_key_value_tree_range_query() {
        let mut tree: KeyValueTree<i32, usize> = KeyValueTree::new();
        tree.add_root(1, 2);
        tree.add_child(1, 2, 3);
        tree.add_child(1, 3, 5);
        tree.add_child(1, 4, 3);
        tree.add_child(1, 5, 1);
        tree.add_child(2, 6, 4);
        tree.add_child(4, 7, 4);
        tree.add_child(4, 8, 3);
        tree.add_child(4, 9, 1);

        let b:(HashMap<i32, usize>, Vec<(usize, usize)>) = tree.dfs_with_subtree_sizes(1);
        let values_in_nodes: Vec<usize> = b.1.iter().map(|&(_, values)| values).collect();
        let segment_tree: Vec<usize> = crate::ranges::build_segment_tree_sum(&values_in_nodes);
        
        let q: i32 = 4;
        let left: usize = b.0[&q];
        let right: usize = left + b.1[left].0 - 1;
        let res: usize = crate::ranges::query_segment_tree_sum(&segment_tree, left, right);

        assert_eq!(res, 11);
    }
}
