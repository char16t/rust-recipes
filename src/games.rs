use std::collections::HashSet;

pub fn nim_sum<T>(elements: &[T]) -> T
where
    T: Default + Copy + std::ops::BitXor<Output = T>,
{
    if elements.len() == 0 {
        return T::default();
    }
    let mut result: T = elements[0];
    for i in 1..elements.len() {
        result = result ^ elements[i];
    }
    result
}

pub fn is_nim_game_over<T>(elements: &[T]) -> bool
where
    T: Default + std::cmp::PartialEq,
{
    for i in 0..elements.len() {
        if elements[i] != T::default() {
            return false;
        }
    }
    return true;
}

/// Return (A, B). It means "Take B elements from A heap"
pub fn nim_move(elements: &[usize]) -> (usize, usize) {
    let mut elem_index: usize = 0;
    let mut n_removed: usize = 0;

    let nim_sum: usize = nim_sum(elements);
    if nim_sum != 0 {
        for (i, &elem) in elements.iter().enumerate() {
            if elem ^ nim_sum < elem {
                elem_index = i;
                n_removed = elem - (elem ^ nim_sum);
                break;
            }
        }
        return (elem_index, n_removed);
    } else {
        for i in 0..elements.len() {
            if elements[i] != 0 {
                return (i, 1);
            }
        }
        return (0, 0);
    }
}

/// Return (A, B). It means "Take B elements from A heap"
pub fn misere_nim_move(elements: &[usize]) -> (usize, usize) {
    let mut elem_index: usize = 0;
    let mut n_removed: usize = 0;

    let mut nim_sum: usize = 0;
    let mut n_heaps_with_size_equals_1: usize = 0;
    let mut n_heaps_with_size_more_than_1: usize = 0;
    let mut heap_with_size_more_than_1: usize = 0;
    for i in 0..elements.len() {
        if elements[i] == 1 {
            n_heaps_with_size_equals_1 += 1;
        }
        if elements[i] > 1 {
            n_heaps_with_size_more_than_1 += 1;
            heap_with_size_more_than_1 = i;
        }
        nim_sum = nim_sum ^ elements[i];
    }

    if n_heaps_with_size_more_than_1 == 1 {
        if n_heaps_with_size_equals_1 != 0 {
            if n_heaps_with_size_equals_1 % 2 == 0 {
                elem_index = heap_with_size_more_than_1;
                n_removed = elements[heap_with_size_more_than_1] - 1;
            } else {
                elem_index = heap_with_size_more_than_1;
                n_removed = elements[heap_with_size_more_than_1];
            }
        } else {
            elem_index = heap_with_size_more_than_1;
            n_removed = elements[heap_with_size_more_than_1] - 1;
        }
        return (elem_index, n_removed);
    }

    if nim_sum != 0 {
        for (i, &elem) in elements.iter().enumerate() {
            if elem ^ nim_sum < elem {
                elem_index = i;
                n_removed = elem - (elem ^ nim_sum);
                break;
            }
        }
        return (elem_index, n_removed);
    } else {
        for i in 0..elements.len() {
            if elements[i] != 0 {
                return (i, 1);
            }
        }
        return (0, 0);
    }
}

pub fn mex(states: &[usize]) -> usize {
    let mut grundy_numbers: HashSet<usize> = HashSet::new();
    for &state in states {
        grundy_numbers.insert(state);
    }
    let mut mex: usize = 0;
    while grundy_numbers.contains(&mex) {
        mex += 1;
    }
    return mex;
}

pub fn is_losing_state(grundy_number: usize) -> bool {
    grundy_number == 0
}

pub fn is_winning_state(grundy_number: usize) -> bool {
    grundy_number != 0
}

pub fn grundy_number(n: usize) -> usize {
    if n == 1 || n == 2 {
        return 0;
    }
    let mut grundy_numbers: Vec<usize> = vec![];
    let mut used_nums: HashSet<usize> = HashSet::new();
    for i in 1..n {
        if n - i != i && !used_nums.contains(&i) {
            grundy_numbers.push(nim_sum(&vec![grundy_number(i), grundy_number(n - i)]));
            used_nums.insert(i);
            used_nums.insert(n - i);
        }
    }
    return mex(&grundy_numbers);
}

/// Return (A, B, C). It means "Split A-th heap into B and C"
pub fn grundy_game_move(heaps: &[usize]) -> (usize, usize, usize) {
    let mut heap_with_more_than_1_element: usize = 0;
    let mut heap_with_more_than_1_element_exists: bool = false;

    for heap_index in 0..heaps.len() {
        let n: usize = heaps[heap_index];

        if n > 1 {
            heap_with_more_than_1_element = heap_index;
            heap_with_more_than_1_element_exists = true;
        }

        let mut used_nums: HashSet<usize> = HashSet::new();
        for i in 1..n {
            if n - i != i && !used_nums.contains(&i) {
                if nim_sum(&vec![grundy_number(i), grundy_number(n - i)]) == 0 {
                    return (heap_index, i, n - i);
                }
                used_nums.insert(i);
                used_nums.insert(n - i);
            }
        }
    }

    if heap_with_more_than_1_element_exists {
        return (
            heap_with_more_than_1_element,
            1,
            heaps[heap_with_more_than_1_element] - 1,
        );
    }
    panic!("Unable to make move when heaps not exists, {:?}", heaps);
}

pub fn is_grundy_game_over(elements: &[usize]) -> bool {
    for i in 0..elements.len() {
        if elements[i] != 1 {
            return false;
        }
    }
    return true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nim_sum() {
        assert_eq!(nim_sum::<i32>(&vec![]), 0);
        assert_eq!(nim_sum(&vec![10, 3]), 9);
        assert_eq!(nim_sum(&vec![1, 4, 5]), 0);
        assert_eq!(nim_sum(&vec![1, 1, 5]), 5);
    }

    #[test]
    fn test_is_nim_game_over() {
        assert_eq!(is_nim_game_over::<i32>(&vec![]), true);
        assert_eq!(is_nim_game_over(&vec![10, 3]), false);
        assert_eq!(is_nim_game_over(&vec![0, 3, 0]), false);
        assert_eq!(is_nim_game_over(&vec![0, 0]), true);
        assert_eq!(is_nim_game_over(&vec![0]), true);
    }

    #[test]
    fn test_nim_move() {
        assert_eq!(nim_move(&vec![]), (0, 0));
        assert_eq!(nim_move(&vec![3, 4, 5]), (0, 2));
        assert_eq!(nim_move(&vec![1, 1, 5]), (2, 5));
        assert_eq!(nim_move(&vec![1, 1, 2]), (2, 2));

        assert_eq!(nim_move(&vec![10, 12, 5]), (0, 1));
        assert_eq!(nim_move(&vec![1, 4, 5]), (0, 1));
    }

    #[test]
    fn test_misere_nim_move() {
        assert_eq!(misere_nim_move(&vec![]), (0, 0));
        assert_eq!(misere_nim_move(&vec![3, 4, 5]), (0, 2));
        assert_eq!(misere_nim_move(&vec![1, 1, 5]), (2, 4));
        assert_eq!(misere_nim_move(&vec![1, 1, 2]), (2, 1));
        assert_eq!(misere_nim_move(&vec![0, 1, 2]), (2, 2));
        assert_eq!(misere_nim_move(&vec![5, 0, 1]), (0, 5));
        assert_eq!(misere_nim_move(&vec![0, 5, 0]), (1, 4));

        assert_eq!(misere_nim_move(&vec![10, 12, 5]), (0, 1));
        assert_eq!(misere_nim_move(&vec![1, 4, 5]), (0, 1));
    }

    #[test]
    fn test_nim_integration() {
        let mut player: bool = true;
        let mut elements: Vec<usize> = vec![3, 4, 5];
        while !is_nim_game_over(&elements) {
            let p: &str = if player { "COMPUTER" } else { "HUMAN" };
            let m: (usize, usize) = nim_move(&elements);
            println!("[{}] Take {} element from {}-th heap", p, m.1, m.0);
            elements[m.0] -= m.1;
            println!("[{}] Elements is {:?}", p, elements);
            player = !player;
        }
        assert_eq!(elements, vec![0, 0, 0]);
        assert_eq!(player, false);
    }

    #[test]
    fn test_misere_nim_integration() {
        let mut player: bool = true;
        let mut elements: Vec<usize> = vec![3, 4, 5];
        while !is_nim_game_over(&elements) {
            let p: &str = if player { "COMPUTER" } else { "HUMAN" };
            let m: (usize, usize) = misere_nim_move(&elements);
            println!("[{}] Take {} element from {}-th heap", p, m.1, m.0);
            elements[m.0] -= m.1;
            println!("[{}] Elements is {:?}", p, elements);
            player = !player;
        }
        assert_eq!(elements, vec![0, 0, 0]);
        assert_eq!(player, true);
    }

    #[test]
    fn test_mex() {
        assert_eq!(mex(&vec![0, 1, 3]), 2);
    }

    #[test]
    fn test_is_losing_state() {
        assert_eq!(is_losing_state(0), true);
        assert_eq!(is_losing_state(5), false);
    }

    #[test]
    fn test_is_winning_state() {
        assert_eq!(is_winning_state(0), false);
        assert_eq!(is_winning_state(5), true);
    }

    #[test]
    fn test_grundy_number() {
        assert_eq!(grundy_number(1), 0);
        assert_eq!(grundy_number(2), 0);
        assert_eq!(grundy_number(3), 1);
        assert_eq!(grundy_number(4), 0);
        assert_eq!(grundy_number(5), 2);
        assert_eq!(grundy_number(6), 1);
        assert_eq!(grundy_number(7), 0);
        assert_eq!(grundy_number(8), 2);
    }

    #[test]
    fn test_grundy_game_move() {
        assert_eq!(grundy_game_move(&vec![8]), (0, 1, 7));
    }

    #[test]
    #[should_panic]
    fn test_grundy_game_move_panic() {
        grundy_game_move(&vec![]);
    }

    #[test]
    fn test_is_grundy_game_over() {
        assert_eq!(is_grundy_game_over(&vec![]), true);
        assert_eq!(is_grundy_game_over(&vec![10, 3]), false);
        assert_eq!(is_grundy_game_over(&vec![1, 3, 1]), false);
        assert_eq!(is_grundy_game_over(&vec![1, 1]), true);
        assert_eq!(is_grundy_game_over(&vec![1]), true);
    }

    #[test]
    fn test_grundy_game_integration() {
        let mut player: bool = true;

        let mut heaps: Vec<usize> = vec![8];
        while !is_grundy_game_over(&heaps) {
            let p: &str = if player { "COMPUTER" } else { "HUMAN" };
            println!("[{}] Elements: {:?}", p, heaps);

            let (heap_index, size_a, size_b) = grundy_game_move(&heaps);
            heaps[heap_index] = size_a;
            heaps.push(size_b);

            println!(
                "[{}] Split: {}-th heap into {} and {}",
                p, heap_index, size_a, size_b
            );
            println!("[{}] Elements: {:?}", p, heaps);
            player = !player;
        }

        assert_eq!(heaps, vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(player, false);
    }
}
