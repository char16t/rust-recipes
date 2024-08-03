pub fn nim_sum<T>(elements: &[T]) -> T
where
    T: Default + Copy + std::ops::BitXor<Output = T>
{
    if elements.len() == 0 {
        return T::default()
    }
    let mut result: T = elements[0];
    for i in 1..elements.len() {
        result = result ^ elements[i];
    }
    result
}

pub fn is_nim_game_over<T>(elements: &[T]) -> bool 
where 
    T: Default + std::cmp::PartialEq
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
    }
    else {
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
    }
    else {
        for i in 0..elements.len() {
            if elements[i] != 0 {
                return (i, 1);
            }
        }
        return (0, 0);
    }
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
}