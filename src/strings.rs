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
}
