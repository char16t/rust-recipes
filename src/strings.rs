use std::collections::HashMap;

use crate::{combinatorics, graphs::AdjacencyListGraph};

pub fn de_bruijn_sequence(alphabet: &[char], length: usize) -> String {
    let vertices: Vec<Vec<char>> = combinatorics::placements_with_repetitions(alphabet, length - 1);
    let mut graph: AdjacencyListGraph<String> = AdjacencyListGraph::new_directed();

    for vertice in vertices {
        let a: String = vertice.iter().collect::<String>();
        for &letter in alphabet {
            let b: String = vertice.iter().skip(1).cloned().chain(std::iter::once(letter)).collect::<String>();
            graph.add_edge(a.clone(), b);
        }
    }

    let result: Vec<String> = graph.eulerian_path_for_connected_graphs();
    let tail: String = result.iter().skip(1).cloned().map(|x| x.chars().last()).filter_map(|x| x).collect();
    
    match result.first() {
        Some(head) => head.clone() + tail.as_str(),
        None => String::new()
    }
}

#[derive(Default, Debug)]
struct TrieNode {
    is_end_of_word: bool,
    children: HashMap<char, TrieNode>,
}

#[derive(Default, Debug)]
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            root: TrieNode::default(),
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut current_node: &mut TrieNode = &mut self.root;

        for c in word.chars() {
            current_node = current_node.children.entry(c).or_default();
        }
        current_node.is_end_of_word = true;
    }

    pub fn contains(&self, word: &str) -> bool {
        let mut current_node = &self.root;

        for c in word.chars() {
            match current_node.children.get(&c) {
                Some(node) => current_node = node,
                None => return false,
            }
        }

        current_node.is_end_of_word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_de_bruijn_sequence() {
        let alphabet: Vec<char> = vec!['0', '1'];
        let n: usize = 3; //length
        let k: usize = alphabet.len();
        let result: String = de_bruijn_sequence(&alphabet, n);
        
        assert_eq!(result.len(), k.pow(n as u32) + n - 1);

        let ps: Vec<Vec<char>> = combinatorics::placements_with_repetitions(&alphabet, 3);
        for p in ps {
            let substr: String = p.iter().collect::<String>();
            assert!(result.contains(&substr))
        }
    }

    #[test]
    fn test_de_bruijn_sequence_fail() {
        let alphabet: Vec<char> = vec![];
        let n: usize = 3; //length
        let _k: usize = alphabet.len();
        let seq: String = de_bruijn_sequence(&alphabet, n);
        assert_eq!(seq, "");
    }

    #[test]
    fn test_trie() {
        let w_latin: &str = "alphabet";
        let w_cyrillic: &str = "алфавит";
        let w_emoji: &str = "🙂🙃😌🥲😁";

        let mut trie: Trie = Trie::new();
        trie.insert(w_latin);
        trie.insert(w_cyrillic);
        trie.insert(w_emoji);

        assert!(trie.contains(w_latin));
        assert!(!trie.contains("alpha"));
        assert!(!trie.contains("alphabetical"));

        assert!(trie.contains(w_cyrillic));
        assert!(!trie.contains("алфа"));
        assert!(!trie.contains("алфавитный"));

        assert!(trie.contains(w_emoji));
        assert!(!trie.contains("🙂🙃"));
        assert!(!trie.contains("🙂🙃😌🥲😁😡🤢🌚"));
    }
}
