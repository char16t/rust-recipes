use std::collections::HashMap;

/// Sequence elements numbering
/// The elements will be assigned numbers in the order of their first occurrence in the sequence.
pub fn enumerate_inplace(a: &mut[i32]) {
    let mut m: HashMap<i32, i32> = HashMap::new();

    for x in a.iter_mut() {
        if let Some(&val) = m.get(x) {
            *x = val;
        } else {
            let size: i32 = m.len() as i32;
            m.insert(*x, size);
            *x = size;
        }
    }
}

/// Sequence elements numbering
/// Preserve the order by assigning smaller numbers to smaller elements.
pub fn enumerate_inplace_ordered(a: &mut [i32]) {
    let mut b: Vec<i32> = a.to_vec();
    b.sort();

    // Fill hashmap
    let mut m: HashMap<i32, i32> = HashMap::new();
    for (idx, &x) in b.iter().enumerate() {
        m.entry(x).or_insert(idx as i32);
    }

    // Compress array
    for x in a.iter_mut() {
        if let Some(&val) = m.get(x) {
            *x = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_inplace() {
        let mut a: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        enumerate_inplace(&mut a);

        assert_eq!(a[0], 0);
        assert_eq!(a[1], 1);
        assert_eq!(a[2], 2);
        assert_eq!(a[3], 1);
        assert_eq!(a[4], 3);
        assert_eq!(a[5], 4);
        assert_eq!(a[6], 5);
        assert_eq!(a[7], 6);
        assert_eq!(a[8], 3);
        assert_eq!(a[9], 0);
        assert_eq!(a[10], 3);
    }

    #[test]
    fn test_enumerate_inplace_ordered() {
        let mut a: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        enumerate_inplace_ordered(&mut a);
        
        assert_eq!(a[0], 3);
        assert_eq!(a[1], 0);
        assert_eq!(a[2], 5);
        assert_eq!(a[3], 0);
        assert_eq!(a[4], 6);
        assert_eq!(a[5], 10);
        assert_eq!(a[6], 2);
        assert_eq!(a[7], 9);
        assert_eq!(a[8], 6);
        assert_eq!(a[9], 3);
        assert_eq!(a[10], 6);
    }
}
