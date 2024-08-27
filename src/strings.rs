use std::{cmp, collections::{HashMap, HashSet, VecDeque}};

use crate::{combinatorics, coordinates, graphs::AdjacencyListGraph};

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

pub fn longest_common_subsequence(a: &str, b: &str) -> String {
    let a_chars: &[u8] = a.as_bytes();
    let b_chars: &[u8] = b.as_bytes();

    let total_rows: usize = a_chars.len() + 1;
    let total_columns: usize = b_chars.len() + 1;
    let xy = coordinates::create_coordinate_function_2d!(total_rows, total_columns);



    let mut table: Vec<usize> = vec![0; total_rows * total_columns];
    for row in 1..total_rows {
        for col in 1..total_columns {
            if a_chars[row - 1] == b_chars[col - 1] {
                table[xy(row, col)] = table[xy(row - 1, col - 1)] + 1;
            } else {
                table[xy(row, col)] = cmp::max(table[xy(row, col - 1)], table[xy(row - 1, col)]);
            }
        }
    }

    let mut common_seq: VecDeque<u8> = VecDeque::new();
    let mut x: usize = total_rows - 1;
    let mut y: usize = total_columns - 1;

    while x != 0 && y != 0 {
        // Check element above is equal
        if table[xy(x, y)] == table[xy(x - 1, y)] {
            x = x - 1;
        }
        // check element to the left is equal
        else if table[xy(x, y)] == table[xy(x, y - 1)] {
            y = y - 1;
        }
        else {
            let char: u8 = a_chars[x - 1];
            common_seq.push_front(char);
            x = x - 1;
            y = y - 1;
        }
    }
    return String::from_utf8(common_seq.into()).unwrap();
}

pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let (aa, bb): (&str, &str) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let w1: Vec<char> = aa.chars().collect::<Vec<char>>();
    let w2: Vec<char> = bb.chars().collect::<Vec<char>>();

    let word1_length: usize = w1.len() + 1;
    let word2_length: usize = w2.len() + 1;

    let mut matrix: Vec<usize> = vec![0; word1_length * word2_length];
    let xy = coordinates::create_coordinate_function_2d!(word2_length, word1_length);

    for i in 1..word1_length { matrix[xy(0, i)] = i; }
    for j in 1..word2_length { matrix[xy(j, 0)] = j; }

    for j in 1..word2_length {
        for i in 1..word1_length {
            let edit_j_i: usize = if w1[i-1] == w2[j-1] {
                matrix[xy(j-1, i-1)]
            } else {
                1 + std::cmp::min(std::cmp::min(matrix[xy(j, i-1)], matrix[xy(j-1, i)]), matrix[xy(j-1, i-1)])
            };
            matrix[xy(j, i)] = edit_j_i;
        }
    }

    matrix[xy(word2_length-1, word1_length-1)]
}

#[allow(dead_code)]
pub struct PolynomialHash {
    h: Vec<usize>, // array of prefix hash-codes
    p: Vec<usize>, // array of A^k mod B
    a: usize,      // const A
    b: usize,      // const B
}

impl PolynomialHash {
    pub fn new(string: &str, a: usize, b: usize) -> Self {
        let bytes: &[u8] = string.as_bytes();
        let n: usize = bytes.len();
        
        let mut h: Vec<usize> = vec![0; n];
        let mut p: Vec<usize> = vec![0; n];
        
        // p[0] = A^0 mod B
        p[0] = 1;

        // h[0] = s[0]
        h[0] = bytes[0] as usize;

        for i in 1..n {
            // h[k] = (h[k - 1] * A + s[k]) mod B
            h[i] = (h[i - 1] * a + bytes[i] as usize) % b;

            // p[k] = (p[k - 1] * A) mod B
            p[i] = (p[i - 1] * a) % b;
        }

        PolynomialHash { h, p, a, b }
    }

    pub fn hash_substring(&self, a: usize, b: usize) -> usize {
        if a == 0 {
            return self.h[b];
        }
        
        let hash_value: usize = (self.h[b] + self.b - (self.h[a - 1] * self.p[b - a + 1] % self.b)) % self.b;

        hash_value
    }
}

pub fn pattern_matching(pattern: &str, string: &str) -> Vec<usize> {
    let mut positions: Vec<usize> = Vec::new();

    let pattern_ph: PolynomialHash = PolynomialHash::new(pattern, 3, 97);
    let string_ph: PolynomialHash = PolynomialHash::new(string, 3, 97);

    let pattern_length: usize = pattern.len();
    let pattern_hash: usize = pattern_ph.hash_substring(0, pattern_length-1);

    for i in 0..string.len()-pattern_length {
        if string_ph.hash_substring(i, i + pattern_length - 1) == pattern_hash {
            positions.push(i);
        }
    }

    positions
}

pub fn count_different_substrings(string: &str, length: usize) -> usize {
    let mut substring_hashes: HashSet<usize> = HashSet::new();

    let string_ph: PolynomialHash = PolynomialHash::new(string, 3, 97);
    for i in 0..string.len()-length {
        substring_hashes.insert(string_ph.hash_substring(i, i + length - 1));
    }

    return substring_hashes.len();
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

    #[test]
    fn test_longest_common_subsequence() {
        let r: String = longest_common_subsequence("abcdef", "fedcba");
        assert_eq!("a", r);

        let r: String = longest_common_subsequence("fedcba", "abcdef");
        assert_eq!("f", r);

        let r: String = longest_common_subsequence("tour", "opera");
        assert_eq!("or", r);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("ABC", "ABCA"), 1);
        assert_eq!(levenshtein_distance("ABC", "AC"), 1);
        assert_eq!(levenshtein_distance("ABC", "ADC"), 1);

        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
    }

    #[test]
    fn test_polynomial_hash() {            
        let string: &str = "ABACB";
        let a: usize = 3;
        let b: usize = 97;

        let ph: PolynomialHash = PolynomialHash::new(string, a, b);

        let hash_value: usize = ph.hash_substring(0, 4);
        assert_eq!(hash_value, 42);

        // Hash for substring "BAC"
        let hash_value_sub: usize = ph.hash_substring(1, 3);
        assert_eq!(hash_value_sub, 80);

        // Hash for string "BAC"
        let hash_value: usize = PolynomialHash::new("BAC", a, b).hash_substring(0, 2);
        assert_eq!(hash_value, 80);
    }

    #[test]
    fn test_pattern_matching() {
        let positions: Vec<usize> = pattern_matching("ABC", "ABCABABCA");
        assert_eq!(positions, vec![0, 5]);
    }

    #[test]
    fn test_count_different_substrings() {
        let c: usize = count_different_substrings("ABABAB", 3);
        assert_eq!(c, 2); // ABA, BAB
    }
}
