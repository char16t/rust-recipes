use std::{collections::HashMap, str::SplitWhitespace};

pub struct WordsBuffer {
    document: String,
    words: HashMap<String, usize>,
}
impl WordsBuffer {
    pub fn new(document: String) -> Self {
        let mut words: HashMap<String, usize> = HashMap::new();
        let ws: SplitWhitespace = document.split_whitespace();
        for word in ws {
            *words.entry(word.to_string()).or_insert(0) += 1;
        }

        Self { document, words }
    }

    pub fn insert(&mut self, word: String) {
        *self.words.entry(word).or_insert(0) += 1;
    }

    pub fn count(&self, word: String) -> usize {
        *self.words.get(&word).unwrap_or(&0)
    }

    pub fn count_all(&self) -> usize {
        let mut sum: usize = 0;
        for (_, &count) in self.words.iter() {
            sum += count;
        }
        sum
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn name(&self) -> String {
        self.document.clone()
    }
}

pub struct WordsBufferIterator<'a> {
    iter: std::collections::hash_map::Iter<'a, String, usize>,
}
impl<'a> Iterator for WordsBufferIterator<'a> {
    type Item = (&'a String, &'a usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a> IntoIterator for &'a WordsBuffer {
    type Item = (&'a String, &'a usize);
    type IntoIter = std::collections::hash_map::Iter<'a, String, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.words.iter()
    }
}

pub fn tf_idf(query: &str, buffers: &[WordsBuffer]) -> HashMap<String, f64> {
    let words: SplitWhitespace = query.split_whitespace();
    let mut scores: HashMap<String, f64> = HashMap::new();
    for word in words {
        let results: Vec<(String, f64)> = tf_idf_word(word, buffers);
        for (name, score) in results.iter() {
            let entry: &mut f64 = scores.entry(name.clone()).or_insert(0.0);
            *entry += score;
        }
    }
    scores
}

pub fn tf_idf_word(word: &str, buffers: &[WordsBuffer]) -> Vec<(String, f64)> {
    let mut results: Vec<(String, f64)> = Vec::new();
    let documents_num: usize = buffers.len();
    let documents_with_term_num: usize = buffers
        .iter()
        .filter(|&b| b.count(word.to_string()) > 0)
        .count();
    let idf: f64 = if documents_with_term_num == 0 {
        0.0
    } else {
        f64::ln(documents_num as f64 / documents_with_term_num as f64)
    };

    for buffer in buffers.iter() {
        let count_all: usize = buffer.count_all();
        let count: usize = buffer.count(word.to_string());
        let tf: f64 = count as f64 / count_all as f64;

        let tf_idf: f64 = tf * idf;
        results.push((buffer.name(), tf_idf));
    }
    results.sort_by(|a, b| b.1.total_cmp(&a.1));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tf_idf_unique() {
        let word: &str = "unique";
        let document: String = "my long text".to_string();
        let words_buffer: WordsBuffer = WordsBuffer::new(document);
        let r: Vec<(String, f64)> = tf_idf_word(word, &[words_buffer]);
        let expected: f64 = 0.0;
        let actual: f64 = r.get(0).unwrap().1;
        assert_eq!(expected, actual);
    }
}
