use crate::numbers;
use std::collections::HashSet;

pub fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let mut result: usize = 1;
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }

    result
}

pub fn factorial_usize(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }

    let mut result: usize = 1;
    for i in 2..=n {
        result *= i;
    }

    result
}

pub fn factorial_i128(n: i128) -> i128 {
    if n <= 1 {
        return 1;
    }

    let mut result: i128 = 1;
    for i in 2..=n {
        result *= i;
    }

    result
}

pub fn multinomial_coefficient(n: usize, k: &[usize]) -> usize {
    let mut denominator: usize = 1;
    for &x in k {
        denominator *= factorial_usize(x);
    }

    factorial_usize(n) / denominator
}

pub fn catalan_number(n: usize) -> usize {
    let mut catalan: Vec<usize> = vec![0; n + 1];
    catalan[0] = 1;

    for i in 1..=n {
        for j in 0..i {
            catalan[i] += catalan[j] * catalan[i - 1 - j];
        }
    }

    catalan[n]
}

pub fn count_combinations(n: usize, k: usize) -> usize {
    binomial_coefficient(n, k)
}

pub fn count_combinations_with_repetitions(n: usize, k: usize) -> usize {
    binomial_coefficient(n + k - 1, k)
}

pub fn count_placements(n: usize, k: usize) -> usize {
    factorial_usize(n) / factorial_usize(n - k)
}

pub fn count_placements_with_repetitions(n: usize, k: usize) -> usize {
    let mut result: usize = 1;
    for _i in 0..k {
        result *= n;
    }
    result
}

pub fn count_permutations(n: usize) -> usize {
    factorial_usize(n)
}

pub fn count_permutations_with_repetitions(n: usize) -> usize {
    count_placements_with_repetitions(n, n)
}

pub fn combinations<T>(elements: &[T], k: usize) -> Vec<Vec<T>>
where
    T: Copy,
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<(Vec<T>, usize, usize)> = Vec::new();
    stack.push((Vec::new(), 0, 0));
    while let Some((mut s, range_start, count)) = stack.pop() {
        if count == k {
            result.push(s);
        } else {
            for (i, _) in elements.iter().enumerate().skip(range_start) {
                s.push(elements[i]);
                stack.push((s.clone(), i + 1, count + 1));
                s.pop();
            }
        }
    }
    result
}

pub fn combinations_with_repetitions<T>(elements: &[T], k: usize) -> Vec<Vec<T>>
where
    T: Copy,
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<(Vec<T>, usize, usize)> = Vec::new();
    stack.push((Vec::new(), 0, 0));
    while let Some((mut s, range_start, count)) = stack.pop() {
        if count == k {
            result.push(s);
        } else {
            for (i, _) in elements.iter().enumerate().skip(range_start) {
                s.push(elements[i]);
                stack.push((s.clone(), i, count + 1)); // Allow repetitions by passing i instead of i + 1
                s.pop();
            }
        }
    }
    result
}

pub fn placements<T>(elements: &[T], k: usize) -> Vec<Vec<T>>
where
    T: Copy,
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<(Vec<T>, Vec<bool>)> = Vec::new();
    stack.push((Vec::new(), vec![false; elements.len()]));
    while let Some((mut s, used)) = stack.pop() {
        if s.len() == k {
            result.push(s);
        } else {
            for i in 0..elements.len() {
                if !used[i] {
                    let mut new_used: Vec<bool> = used.clone();
                    new_used[i] = true;
                    s.push(elements[i]);
                    stack.push((s.clone(), new_used));
                    s.pop();
                }
            }
        }
    }
    result
}

pub fn placements_with_repetitions<T>(elements: &[T], k: usize) -> Vec<Vec<T>>
where
    T: Copy,
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<Vec<T>> = Vec::new();
    stack.push(Vec::new());
    while let Some(mut s) = stack.pop() {
        if s.len() == k {
            result.push(s);
        } else {
            for &element in elements.iter() {
                s.push(element);
                stack.push(s.clone());
                s.pop();
            }
        }
    }
    result
}

pub fn permutations<T>(elements: &[T]) -> Vec<Vec<T>>
where
    T: Copy,
{
    placements(elements, elements.len())
}

pub fn permutations_with_repetitions<T>(elements: &[T]) -> Vec<Vec<T>>
where
    T: Copy,
{
    placements_with_repetitions(elements, elements.len())
}

pub fn inclusion_exclusion<R>(unions: &[HashSet<R>]) -> isize
where
    R: Copy + Eq + std::hash::Hash,
{
    let mut result: isize = 0;
    let mut sign: isize = 1;
    for ki in 1..=unions.len() {
        let mut stack: Vec<(Vec<HashSet<R>>, usize, usize)> = Vec::new();
        stack.push((Vec::new(), 0, 0));
        while let Some((mut s, range_start, count)) = stack.pop() {
            if count == ki {
                let union: HashSet<R> = s.iter().flat_map(|set| set.iter()).cloned().collect();
                result += sign * (union.len() as isize);
            } else {
                for (i, union) in unions.iter().enumerate().skip(range_start) {
                    s.push(union.clone());
                    stack.push((s.clone(), i + 1, count + 1));
                    s.pop();
                }
            }
        }
        sign *= -1;
    }
    result
}

pub fn count_derangements(n: usize) -> usize {
    match n {
        1 => 0,
        2 => 1,
        _ => {
            let mut prev = 0;
            let mut current = 1;

            for _ in 3..=n {
                let next = (n - 1) * (prev + current);
                prev = current;
                current = next;
            }

            current
        }
    }
}

pub fn burnsides_lemma(n: usize, m: usize) -> usize {
    let mut sum: usize = 0;
    for k in 0..n {
        let gcd: usize = numbers::gcd(k, n);
        let mut pow: usize = 1;
        for _ in 0..gcd {
            pow *= m;
        }
        sum += pow;
    }
    sum / n
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_binomial_coefficient() {
        let n: usize = 5;
        let k: usize = 3;
        let r: usize = binomial_coefficient(n, k);
        assert_eq!(r, 10);

        let n: usize = 3;
        let k: usize = 3;
        let r: usize = binomial_coefficient(n, k);
        assert_eq!(r, 1);

        let n: usize = 3;
        let k: usize = 5;
        let r: usize = binomial_coefficient(n, k);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_factorial_usize() {
        assert_eq!(factorial_usize(0), 1);
        assert_eq!(factorial_usize(1), 1);
        assert_eq!(factorial_usize(2), 2);
        assert_eq!(factorial_usize(3), 6);
        assert_eq!(factorial_usize(10), 3628800);
    }

    #[test]
    fn test_factorial_i128() {
        assert_eq!(factorial_i128(0), 1);
        assert_eq!(factorial_i128(1), 1);
        assert_eq!(factorial_i128(2), 2);
        assert_eq!(factorial_i128(3), 6);
        assert_eq!(factorial_i128(10), 3628800);
    }

    #[test]
    fn test_multinomial_coefficient() {
        let r: usize = multinomial_coefficient(3, &[2, 0, 1]);
        assert_eq!(r, 3);

        let r: usize = multinomial_coefficient(3, &[1, 1, 1]);
        assert_eq!(r, 6);
    }

    #[test]
    fn test_catalan_number() {
        assert_eq!(catalan_number(0), 1);
        assert_eq!(catalan_number(3), 5);
    }

    #[test]
    fn test_combinations() {
        let r: Vec<Vec<char>> = combinations(&['A', 'B', 'C'], 2);
        assert_eq!(r, vec![vec!['B', 'C'], vec!['A', 'C'], vec!['A', 'B']]);

        let r: Vec<Vec<i16>> = combinations(&[1, 2, 3, 4, 5, 6], 3);
        assert_eq!(
            r,
            vec![
                vec![4, 5, 6],
                vec![3, 5, 6],
                vec![3, 4, 6],
                vec![3, 4, 5],
                vec![2, 5, 6],
                vec![2, 4, 6],
                vec![2, 4, 5],
                vec![2, 3, 6],
                vec![2, 3, 5],
                vec![2, 3, 4],
                vec![1, 5, 6],
                vec![1, 4, 6],
                vec![1, 4, 5],
                vec![1, 3, 6],
                vec![1, 3, 5],
                vec![1, 3, 4],
                vec![1, 2, 6],
                vec![1, 2, 5],
                vec![1, 2, 4],
                vec![1, 2, 3]
            ]
        );

        let r: Vec<Vec<char>> = combinations(&['A', 'B', 'C'], 3);
        assert_eq!(r, vec![vec!['A', 'B', 'C']]);
    }

    #[test]
    fn test_combinations_with_repetitions() {
        let r: Vec<Vec<char>> = combinations_with_repetitions(&['A', 'B', 'C'], 2);
        assert_eq!(
            r,
            vec![
                vec!['C', 'C'],
                vec!['B', 'C'],
                vec!['B', 'B'],
                vec!['A', 'C'],
                vec!['A', 'B'],
                vec!['A', 'A']
            ]
        );
    }

    #[test]
    fn test_placements() {
        let r: Vec<Vec<char>> = placements(&['A', 'B', 'C'], 2);
        assert_eq!(
            r,
            vec![
                vec!['C', 'B'],
                vec!['C', 'A'],
                vec!['B', 'C'],
                vec!['B', 'A'],
                vec!['A', 'C'],
                vec!['A', 'B']
            ]
        );

        let r: Vec<Vec<char>> = placements(&['A', 'B', 'C'], 3);
        assert_eq!(
            r,
            vec![
                vec!['C', 'B', 'A'],
                vec!['C', 'A', 'B'],
                vec!['B', 'C', 'A'],
                vec!['B', 'A', 'C'],
                vec!['A', 'C', 'B'],
                vec!['A', 'B', 'C']
            ]
        );
    }

    #[test]
    fn test_placements_with_repetitions() {
        let r: Vec<Vec<char>> = placements_with_repetitions(&['A', 'B', 'C'], 2);
        assert_eq!(
            r,
            vec![
                vec!['C', 'C'],
                vec!['C', 'B'],
                vec!['C', 'A'],
                vec!['B', 'C'],
                vec!['B', 'B'],
                vec!['B', 'A'],
                vec!['A', 'C'],
                vec!['A', 'B'],
                vec!['A', 'A']
            ]
        );
    }

    #[test]
    fn test_permutations() {
        let p: Vec<Vec<char>> = permutations(&['A', 'B', 'C']);
        assert_eq!(
            p,
            vec![
                vec!['C', 'B', 'A'],
                vec!['C', 'A', 'B'],
                vec!['B', 'C', 'A'],
                vec!['B', 'A', 'C'],
                vec!['A', 'C', 'B'],
                vec!['A', 'B', 'C']
            ]
        );
    }

    #[test]
    fn test_permutations_with_repetitions() {
        let p: Vec<Vec<char>> = permutations_with_repetitions(&['A', 'B', 'C']);
        assert_eq!(
            p,
            vec![
                vec!['C', 'C', 'C'],
                vec!['C', 'C', 'B'],
                vec!['C', 'C', 'A'],
                vec!['C', 'B', 'C'],
                vec!['C', 'B', 'B'],
                vec!['C', 'B', 'A'],
                vec!['C', 'A', 'C'],
                vec!['C', 'A', 'B'],
                vec!['C', 'A', 'A'],
                vec!['B', 'C', 'C'],
                vec!['B', 'C', 'B'],
                vec!['B', 'C', 'A'],
                vec!['B', 'B', 'C'],
                vec!['B', 'B', 'B'],
                vec!['B', 'B', 'A'],
                vec!['B', 'A', 'C'],
                vec!['B', 'A', 'B'],
                vec!['B', 'A', 'A'],
                vec!['A', 'C', 'C'],
                vec!['A', 'C', 'B'],
                vec!['A', 'C', 'A'],
                vec!['A', 'B', 'C'],
                vec!['A', 'B', 'B'],
                vec!['A', 'B', 'A'],
                vec!['A', 'A', 'C'],
                vec!['A', 'A', 'B'],
                vec!['A', 'A', 'A']
            ]
        );
    }

    #[test]
    fn test_inclusion_exclusion() {
        let a: HashSet<i32> = HashSet::from_iter(vec![4]);
        let b: HashSet<i32> = HashSet::from_iter(vec![2]);
        let c: HashSet<i32> = HashSet::from_iter(vec![3]);
        let r = inclusion_exclusion(&[a, b, c]);
        assert_eq!(r, 0);

        let a: HashSet<i32> = HashSet::from_iter(vec![1, 4]);
        let b: HashSet<i32> = HashSet::from_iter(vec![1, 2]);
        let c: HashSet<i32> = HashSet::from_iter(vec![1, 3]);
        let r = inclusion_exclusion(&[a, b, c]);
        assert_eq!(r, 1);

        let a: HashSet<i32> = HashSet::from_iter(vec![2, 7]);
        let b: HashSet<i32> = HashSet::from_iter(vec![2, 3]);
        let c: HashSet<i32> = HashSet::from_iter(vec![3, 4]);
        let r = inclusion_exclusion(&[a, b, c]);
        assert_eq!(r, 0);

        let a: HashSet<i32> = HashSet::from_iter(vec![1, 2, 3]);
        let b: HashSet<i32> = HashSet::from_iter(vec![2, 3, 4]);
        let c: HashSet<i32> = HashSet::from_iter(vec![3, 4, 5]);
        let r = inclusion_exclusion(&[a, b, c]);
        assert_eq!(r, 1);
    }

    #[test]
    fn test_count_derangements() {
        assert_eq!(count_derangements(1), 0);
        assert_eq!(count_derangements(2), 1);
        assert_eq!(count_derangements(3), 2);
    }

    #[test]
    fn test_burnsides_lemma() {
        assert_eq!(burnsides_lemma(4, 3), 24);
    }

    #[test]
    fn test_count_combinations() {
        let n: usize = 5;
        let k: usize = 3;
        let r: usize = count_combinations(n, k);
        let comb: Vec<Vec<i32>> = combinations(&[1, 2, 3, 4, 5], 3);
        assert_eq!(r, 10);
        assert_eq!(r, comb.len());

        let n: usize = 3;
        let k: usize = 3;
        let r: usize = count_combinations(n, k);
        let comb: Vec<Vec<i32>> = combinations(&[1, 2, 3], 3);
        assert_eq!(r, 1);
        assert_eq!(r, comb.len());

        let n: usize = 3;
        let k: usize = 5;
        let r: usize = count_combinations(n, k);
        let comb: Vec<Vec<i32>> = combinations(&[1, 2, 3], 5);
        assert_eq!(r, 0);
        assert_eq!(r, comb.len());
    }

    #[test]
    fn test_count_combinations_with_repetitions() {
        let n: usize = 5;
        let k: usize = 3;
        let r: usize = count_combinations_with_repetitions(n, k);
        let combinations: Vec<Vec<i32>> = combinations_with_repetitions(&[1, 2, 3, 4, 5], 3);
        assert_eq!(r, 35);
        assert_eq!(r, combinations.len());

        let n: usize = 3;
        let k: usize = 3;
        let r: usize = count_combinations_with_repetitions(n, k);
        let combinations: Vec<Vec<i32>> = combinations_with_repetitions(&[1, 2, 3], 3);
        assert_eq!(r, combinations.len());
        assert_eq!(r, 10);

        let n: usize = 3;
        let k: usize = 5;
        let r: usize = count_combinations_with_repetitions(n, k);
        let combinations: Vec<Vec<i32>> = combinations_with_repetitions(&[1, 2, 3], 5);
        assert_eq!(r, combinations.len());
    }

    #[test]
    fn test_count_placements() {
        let r: Vec<Vec<char>> = placements(&['A', 'B', 'C'], 2);
        let c: usize = count_placements(3, 2);
        assert_eq!(c, 6);
        assert_eq!(c, r.len());

        let r: Vec<Vec<char>> = placements(&['A', 'B', 'C'], 3);
        let c: usize = count_placements(3, 3);
        assert_eq!(c, 6);
        assert_eq!(c, r.len());
    }

    #[test]
    fn test_count_placements_with_repetitions() {
        let r: Vec<Vec<char>> = placements_with_repetitions(&['A', 'B', 'C'], 2);
        let c: usize = count_placements_with_repetitions(3, 2);
        assert_eq!(c, 9);
        assert_eq!(c, r.len());

        let r: Vec<Vec<char>> = placements_with_repetitions(&['A', 'B', 'C'], 3);
        let c: usize = count_placements_with_repetitions(3, 3);
        assert_eq!(c, 27);
        assert_eq!(c, r.len());
    }

    #[test]
    fn test_count_permutations() {
        let c: usize = count_permutations(3);
        let p: Vec<Vec<char>> = permutations(&['A', 'B', 'C']);
        assert_eq!(c, 6);
        assert_eq!(c, p.len());
    }

    #[test]
    fn test_count_permutations_with_repetitions() {
        let c: usize = count_permutations_with_repetitions(3);
        let p: Vec<Vec<char>> = permutations_with_repetitions(&['A', 'B', 'C']);
        assert_eq!(c, 27);
        assert_eq!(c, p.len());
    }
}
