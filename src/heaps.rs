pub struct MinBinaryHeap<T> {
    pub data: Vec<T>,
    capacity: usize,
    pub size: usize
}

impl<T> MinBinaryHeap<T>
where
    T: Default + Copy + PartialOrd
{
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            size: 0
        }
    }
    pub fn insert(&mut self, value: T) -> Result<(), ()> {
        if self.size >= self.capacity {
            return Err(());
        }
        self.data[self.size] = value;
        let mut current_position: usize = self.size;
        let mut parent_position: usize = self.parent(current_position);
        let mut current: T = self.data[current_position];
        let mut parent: T = self.data[parent_position];
        while current < parent {
            self.swap(current_position, parent_position);
            current_position = parent_position;
            parent_position = self.parent(current_position);
            current = self.data[current_position];
            parent = self.data[parent_position];
        }
        self.size += 1;
        Ok(())
    }
    
    pub fn remove(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let peak: T = self.data[0];
        self.size -= 1;
        self.swap(0, self.size);
        self.min_heapify();
        return Some(peak);
    }
    
    #[inline(always)]
    fn min_heapify(&mut self) {
        let mut current_position: usize = 0;
        if self.is_leaf(current_position) {
            return;
        }
        let mut left_position: usize = self.left(current_position);
        let mut right_position: usize = self.right(current_position);
        let mut current: T = self.data[current_position];
        let mut left: T = self.data[left_position];
        let mut right: T = self.data[right_position];

        let size: usize = self.size;
        while (current > left && left_position < size) || (current > right && right_position < size) {
            let swap_position: usize = if left < right {
                if left_position < size {
                    left_position
                } else {
                    right_position
                }
            } else {
                if right_position < size {
                    right_position
                } else {
                    left_position
                }
            };
            self.swap(current_position, swap_position);
            current_position = swap_position;
            if self.is_leaf(current_position) {
                break;
            }
            left_position = self.left(current_position);
            right_position = self.right(current_position);
            current = self.data[current_position];
            left = self.data[left_position];
            right = self.data[right_position];
        }
    }

    #[inline(always)]
    fn parent(&self, pos: usize) -> usize {
        if pos > 0 { (pos - 1) / 2 } else { 0 }
    }

    #[inline(always)]
    fn left(&self, pos: usize) -> usize {
        (2 * pos) + 1
    }

    #[inline(always)]
    fn right(&self, pos: usize) -> usize {
        (2 * pos) + 2
    }
    
    #[inline(always)]
    fn is_leaf(&self, pos: usize) -> bool {
        pos > self.size / 2
    }

    #[inline(always)]
    fn swap(&mut self, first: usize, second: usize) {
        let buffer: T = self.data[first];
        self.data[first] = self.data[second];
        self.data[second] = buffer;
    }
}

pub struct MaxBinaryHeap<T> {
    pub data: Vec<T>,
    capacity: usize,
    pub size: usize
}

impl<T> MaxBinaryHeap<T>
where
    T: Default + Copy + PartialOrd
{
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            size: 0
        }
    }
    pub fn insert(&mut self, value: T) -> Result<(), ()> {
        if self.size >= self.capacity {
            return Err(());
        }
        self.data[self.size] = value;
        let mut current_position: usize = self.size;
        let mut parent_position: usize = self.parent(current_position);
        let mut current: T = self.data[current_position];
        let mut parent: T = self.data[parent_position];
        while current > parent {
            self.swap(current_position, parent_position);
            current_position = parent_position;
            parent_position = self.parent(current_position);
            current = self.data[current_position];
            parent = self.data[parent_position];
        }
        self.size += 1;
        Ok(())
    }
    
    pub fn remove(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let peak: T = self.data[0];
        self.size -= 1;
        self.swap(0, self.size);
        self.max_heapify();
        return Some(peak);
    }
    
    #[inline(always)]
    fn max_heapify(&mut self) {
        let mut current_position: usize = 0;
        if self.is_leaf(current_position) {
            return;
        }
        let mut left_position: usize = self.left(current_position);
        let mut right_position: usize = self.right(current_position);
        let mut current: T = self.data[current_position];
        let mut left: T = self.data[left_position];
        let mut right: T = self.data[right_position];

        let size: usize = self.size;
        while (current < left && left_position < size) || (current < right && right_position < size) {
            let swap_position: usize = if left > right {
                if left_position < size {
                    left_position
                } else {
                    right_position
                }
            } else {
                if right_position < size {
                    right_position
                } else {
                    left_position
                }
            };
            self.swap(current_position, swap_position);
            current_position = swap_position;
            if self.is_leaf(current_position) {
                break;
            }
            left_position = self.left(current_position);
            right_position = self.right(current_position);
            current = self.data[current_position];
            left = self.data[left_position];
            right = self.data[right_position];
        }
    }

    #[inline(always)]
    fn parent(&self, pos: usize) -> usize {
        if pos > 0 { (pos - 1) / 2 } else { 0 }
    }

    #[inline(always)]
    fn left(&self, pos: usize) -> usize {
        (2 * pos) + 1
    }

    #[inline(always)]
    fn right(&self, pos: usize) -> usize {
        (2 * pos) + 2
    }
    
    #[inline(always)]
    fn is_leaf(&self, pos: usize) -> bool {
        pos > self.size / 2
    }

    #[inline(always)]
    fn swap(&mut self, first: usize, second: usize) {
        let buffer: T = self.data[first];
        self.data[first] = self.data[second];
        self.data[second] = buffer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(unused_must_use)]
    fn test_min_binary_heap() {
        let mut min_heap: MinBinaryHeap<i32> = MinBinaryHeap::with_capacity(3);
        assert_eq!(min_heap.insert(4), Ok(()));
        assert_eq!(min_heap.insert(5), Ok(()));
        assert_eq!(min_heap.insert(6), Ok(()));
        assert_eq!(min_heap.insert(7), Err(()));

        let mut min_heap: MinBinaryHeap<i32> = MinBinaryHeap::with_capacity(10);
        assert_eq!(min_heap.remove(), None);
        
        min_heap.insert(1);
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), None);

        min_heap.insert(1);
        min_heap.insert(2);
        min_heap.insert(3);
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(3));

        min_heap.insert(3);
        min_heap.insert(2);
        min_heap.insert(1);
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(3));

        min_heap.insert(2);
        min_heap.insert(5);
        min_heap.insert(1);
        min_heap.insert(2);
        min_heap.insert(3);
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(5));

        min_heap.insert(3);
        min_heap.insert(7);
        min_heap.insert(2);
        min_heap.insert(1);
        min_heap.insert(3);
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(7));

        min_heap.insert(1000);
        min_heap.insert(2);
        min_heap.insert(3);
        min_heap.insert(1);
        min_heap.insert(-1000);
        assert_eq!(min_heap.remove(), Some(-1000));

        min_heap.insert(-10);
        min_heap.insert(2000);

        assert_eq!(min_heap.remove(), Some(-10));

        min_heap.insert(-2000);
        assert_eq!(min_heap.remove(), Some(-2000));

        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(1000));
        assert_eq!(min_heap.remove(), Some(2000));

        min_heap.insert(-2000);
        min_heap.insert(-3000);
        min_heap.insert(-4000);
        assert_eq!(min_heap.remove(), Some(-4000));
        assert_eq!(min_heap.remove(), Some(-3000));
        assert_eq!(min_heap.remove(), Some(-2000));
    }

    #[test]
    #[allow(unused_must_use)]
    fn test_max_binary_heap() {
        let mut min_heap: MaxBinaryHeap<i32> = MaxBinaryHeap::with_capacity(3);
        assert_eq!(min_heap.insert(4), Ok(()));
        assert_eq!(min_heap.insert(5), Ok(()));
        assert_eq!(min_heap.insert(6), Ok(()));
        assert_eq!(min_heap.insert(7), Err(()));

        let mut min_heap: MaxBinaryHeap<i32> = MaxBinaryHeap::with_capacity(10);
        assert_eq!(min_heap.remove(), None);
        
        min_heap.insert(1);
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), None);

        min_heap.insert(1);
        min_heap.insert(2);
        min_heap.insert(3);
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(1));

        min_heap.insert(3);
        min_heap.insert(2);
        min_heap.insert(1);
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(1));

        min_heap.insert(2);
        min_heap.insert(5);
        min_heap.insert(1);
        min_heap.insert(2);
        min_heap.insert(3);
        assert_eq!(min_heap.remove(), Some(5));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(1));

        min_heap.insert(3);
        min_heap.insert(7);
        min_heap.insert(2);
        min_heap.insert(1);
        min_heap.insert(3);
        assert_eq!(min_heap.remove(), Some(7));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(3));
        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(1));

        min_heap.insert(1000);
        min_heap.insert(2);
        min_heap.insert(3);
        min_heap.insert(1);
        min_heap.insert(-1000);
        assert_eq!(min_heap.remove(), Some(1000));

        min_heap.insert(-10);
        min_heap.insert(2000);

        assert_eq!(min_heap.remove(), Some(2000));

        min_heap.insert(-2000);
        assert_eq!(min_heap.remove(), Some(3));

        assert_eq!(min_heap.remove(), Some(2));
        assert_eq!(min_heap.remove(), Some(1));
        assert_eq!(min_heap.remove(), Some(-10));
        assert_eq!(min_heap.remove(), Some(-1000));
        assert_eq!(min_heap.remove(), Some(-2000));

        min_heap.insert(-2000);
        min_heap.insert(-3000);
        min_heap.insert(-4000);
        assert_eq!(min_heap.remove(), Some(-2000));
        assert_eq!(min_heap.remove(), Some(-3000));
        assert_eq!(min_heap.remove(), Some(-4000));
    }
}