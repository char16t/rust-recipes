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

pub fn combinations<T>(elements: &[T], k: usize) -> Vec<Vec<T>>
where 
    T: Copy
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<(Vec<T>, usize, usize)> = Vec::new();
    stack.push((Vec::new(), 0, 0));
    while let Some((mut s, range_start, count)) = stack.pop() {
        if count == k {
            result.push(s);
        } else {
            for i in range_start..elements.len() {
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
    T: Copy
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<(Vec<T>, usize, usize)> = Vec::new();
    stack.push((Vec::new(), 0, 0));
    while let Some((mut s, range_start, count)) = stack.pop() {
        if count == k {
            result.push(s);
        } else {
            for i in range_start..elements.len() {
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
    T: Copy
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
    T: Copy
{
    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stack: Vec<Vec<T>> = Vec::new();
    stack.push(Vec::new());
    while let Some(mut s) = stack.pop() {
        if s.len() == k {
            result.push(s);
        } else {
            for i in 0..elements.len() {
                s.push(elements[i]);
                stack.push(s.clone());
                s.pop();
            }
        }
    }
    result
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
        let r: usize = multinomial_coefficient(3, &vec![2, 0, 1]);
        assert_eq!(r, 3);

        let r: usize = multinomial_coefficient(3, &vec![1, 1, 1]);
        assert_eq!(r, 6);
    }

    #[test]
    fn test_catalan_number() {
        assert_eq!(catalan_number(0), 1);
        assert_eq!(catalan_number(3), 5);
    }

    #[test]
    fn test_combinations() {
        let r: Vec<Vec<char>> = combinations(&vec!['A', 'B', 'C'], 2);
        assert_eq!(r, vec![vec!['B', 'C'], vec!['A', 'C'], vec!['A', 'B']]);

        let r: Vec<Vec<i16>> = combinations(&vec![1, 2, 3, 4, 5, 6], 3);
        assert_eq!(r, 
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

        let r: Vec<Vec<char>> = combinations(&vec!['A', 'B', 'C'], 3);
        assert_eq!(r, vec![vec!['A', 'B', 'C']]);
    }

    #[test]
    fn test_combinations_with_repetitions() {
        let r: Vec<Vec<char>> = combinations_with_repetitions(&vec!['A', 'B', 'C'], 2);
        assert_eq!(r, vec![
            vec!['C', 'C'], 
            vec!['B', 'C'], 
            vec!['B', 'B'], 
            vec!['A', 'C'], 
            vec!['A', 'B'], 
            vec!['A', 'A']
        ]);
    }

    #[test]
    fn test_placements() {
        let r: Vec<Vec<char>> = placements(&vec!['A', 'B', 'C'], 2);
        assert_eq!(r, vec![
            vec!['C', 'B'], 
            vec!['C', 'A'], 
            vec!['B', 'C'], 
            vec!['B', 'A'], 
            vec!['A', 'C'], 
            vec!['A', 'B']
        ]);

        let r: Vec<Vec<char>> = placements(&vec!['A', 'B', 'C'], 3);
        assert_eq!(r, vec![
            vec!['C', 'B', 'A'], 
            vec!['C', 'A', 'B'], 
            vec!['B', 'C', 'A'], 
            vec!['B', 'A', 'C'], 
            vec!['A', 'C', 'B'], 
            vec!['A', 'B', 'C']
        ]);
    }

    #[test]
    fn test_placements_with_repetitions() {
        let r: Vec<Vec<char>> = placements_with_repetitions(&vec!['A', 'B', 'C'], 2);
        assert_eq!(r, vec![
            vec!['C', 'C'], 
            vec!['C', 'B'], 
            vec!['C', 'A'], 
            vec!['B', 'C'], 
            vec!['B', 'B'], 
            vec!['B', 'A'], 
            vec!['A', 'C'], 
            vec!['A', 'B'], 
            vec!['A', 'A']
        ]);
    }
}
