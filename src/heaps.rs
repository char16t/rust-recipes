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
        let mut data: Vec<T> = Vec::with_capacity(0);
        data.resize(capacity, T::default());
        Self {
            data,
            capacity,
            size: 0
        }
    }
    pub fn insert(&mut self, value: T) {
        if self.size >= self.capacity {
            self.data.resize(self.capacity * 2 + 1, T::default());
            self.capacity *= 2;
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

pub struct MinBinaryHeap2<T, W> {
    pub data: Vec<(T, W)>,
    capacity: usize,
    pub size: usize
}

impl<T, W> MinBinaryHeap2<T, W>
where
    T: Default + Copy,
    W: Default + Copy + PartialOrd
{
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data: Vec<(T, W)> = Vec::with_capacity(0);
        data.resize(capacity, (T::default(), W::default()));
        Self {
            data,
            capacity,
            size: 0
        }
    }
    pub fn insert(&mut self, value: T, weight: W) {
        if self.size >= self.capacity {
            self.data.resize(self.capacity * 2 + 1, (T::default(), W::default()));
            self.capacity *= 2;
        }
        self.data[self.size] = (value, weight);
        let mut current_position: usize = self.size;
        let mut parent_position: usize = self.parent(current_position);
        let mut current: (T, W) = self.data[current_position];
        let mut parent: (T, W) = self.data[parent_position];

        while current.1 < parent.1 {
            self.swap(current_position, parent_position);
            current_position = parent_position;
            parent_position = self.parent(current_position);
            current = self.data[current_position];
            parent = self.data[parent_position];
        }
        self.size += 1;
    }
    
    pub fn remove(&mut self) -> Option<(T, W)> {
        if self.size == 0 {
            return None;
        }
        let peak: (T, W) = self.data[0];
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
        let mut current: (T, W) = self.data[current_position];
        let mut left: (T, W) = self.data[left_position];
        let mut right: (T, W) = self.data[right_position];

        let size: usize = self.size;
        while (current.1 > left.1 && left_position < size) || (current.1 > right.1 && right_position < size) {
            let swap_position: usize = if left.1 < right.1 {
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
        let buffer: (T, W) = self.data[first];
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
        let mut data: Vec<T> = Vec::with_capacity(0);
        data.resize(capacity, T::default());
        Self {
            data,
            capacity,
            size: 0
        }
    }
    pub fn insert(&mut self, value: T) {
        if self.size >= self.capacity {
            self.data.resize(self.capacity * 2 + 1, T::default());
            self.capacity *= 2;
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

pub struct MaxBinaryHeap2<T, W> {
    pub data: Vec<(T, W)>,
    capacity: usize,
    pub size: usize
}

impl<T, W> MaxBinaryHeap2<T, W>
where
    T: Default + Copy,
    W: Default + Copy + PartialOrd
{
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data: Vec<(T, W)> = Vec::with_capacity(0);
        data.resize(capacity, (T::default(), W::default()));
        Self {
            data: vec![(T::default(), W::default()); capacity],
            capacity,
            size: 0
        }
    }
    pub fn insert(&mut self, value: T, weight: W) {
        if self.size >= self.capacity {
            self.data.resize(self.capacity * 2 + 1, (T::default(), W::default()));
            self.capacity *= 2;
        }
        self.data[self.size] = (value, weight);
        let mut current_position: usize = self.size;
        let mut parent_position: usize = self.parent(current_position);
        let mut current: (T, W) = self.data[current_position];
        let mut parent: (T, W) = self.data[parent_position];
        while current.1 > parent.1 {
            self.swap(current_position, parent_position);
            current_position = parent_position;
            parent_position = self.parent(current_position);
            current = self.data[current_position];
            parent = self.data[parent_position];
        }
        self.size += 1;
    }
    
    pub fn remove(&mut self) -> Option<(T, W)> {
        if self.size == 0 {
            return None;
        }
        let peak: (T, W) = self.data[0];
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
        let mut current: (T, W) = self.data[current_position];
        let mut left: (T, W) = self.data[left_position];
        let mut right: (T, W) = self.data[right_position];

        let size: usize = self.size;
        while (current.1 < left.1 && left_position < size) || (current.1 < right.1 && right_position < size) {
            let swap_position: usize = if left.1 > right.1 {
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
        let buffer: (T, W) = self.data[first];
        self.data[first] = self.data[second];
        self.data[second] = buffer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_binary_heap() {
        let mut min_heap: MinBinaryHeap<i32> = MinBinaryHeap::with_capacity(3);
        min_heap.insert(4);
        min_heap.insert(5);
        min_heap.insert(6);
        assert_eq!(min_heap.capacity, 3);
        min_heap.insert(7);
        min_heap.insert(8);
        min_heap.insert(9);
        assert_eq!(min_heap.capacity, 6);
        assert_eq!(min_heap.remove(), Some(4));
        assert_eq!(min_heap.remove(), Some(5));
        assert_eq!(min_heap.remove(), Some(6));
        assert_eq!(min_heap.remove(), Some(7));
        assert_eq!(min_heap.remove(), Some(8));
        assert_eq!(min_heap.remove(), Some(9));        

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
    fn test_min_binary_heap_2() {
        let mut min_heap: MinBinaryHeap2<char, i64> = MinBinaryHeap2::with_capacity(3);
        min_heap.insert('A', 4);
        min_heap.insert('B', 5);
        min_heap.insert('C', 6);
        assert_eq!(min_heap.capacity, 3);
        min_heap.insert('D', 7);
        min_heap.insert('E', 8);
        min_heap.insert('F', 9);
        assert_eq!(min_heap.capacity, 6);
        assert_eq!(min_heap.remove(), Some(('A', 4)));
        assert_eq!(min_heap.remove(), Some(('B', 5)));
        assert_eq!(min_heap.remove(), Some(('C', 6)));
        assert_eq!(min_heap.remove(), Some(('D', 7)));
        assert_eq!(min_heap.remove(), Some(('E', 8)));
        assert_eq!(min_heap.remove(), Some(('F', 9)));        

        let mut min_heap: MinBinaryHeap2<char, i64> = MinBinaryHeap2::with_capacity(10);
        assert_eq!(min_heap.remove(), None);
        
        min_heap.insert('A', 1);
        assert_eq!(min_heap.remove(), Some(('A', 1)));
        assert_eq!(min_heap.remove(), None);

        min_heap.insert('A', 1);
        min_heap.insert('B', 2);
        min_heap.insert('C', 3);
        assert_eq!(min_heap.remove(), Some(('A', 1)));
        assert_eq!(min_heap.remove(), Some(('B', 2)));
        assert_eq!(min_heap.remove(), Some(('C', 3)));

        min_heap.insert('A', 3);
        min_heap.insert('B', 2);
        min_heap.insert('C', 1);
        assert_eq!(min_heap.remove(), Some(('C', 1)));
        assert_eq!(min_heap.remove(), Some(('B', 2)));
        assert_eq!(min_heap.remove(), Some(('A', 3)));

        min_heap.insert('A', 2);
        min_heap.insert('B', 5);
        min_heap.insert('C', 1);
        min_heap.insert('D', 2);
        min_heap.insert('E', 3);
        assert_eq!(min_heap.remove(), Some(('C', 1)));
        assert_eq!(min_heap.remove(), Some(('A', 2)));
        assert_eq!(min_heap.remove(), Some(('D', 2)));
        assert_eq!(min_heap.remove(), Some(('E', 3)));
        assert_eq!(min_heap.remove(), Some(('B', 5)));

        min_heap.insert('A', 3);
        min_heap.insert('B', 7);
        min_heap.insert('C', 2);
        min_heap.insert('D', 1);
        min_heap.insert('E', 3);
        assert_eq!(min_heap.remove(), Some(('D', 1)));
        assert_eq!(min_heap.remove(), Some(('C', 2)));
        assert_eq!(min_heap.remove(), Some(('A', 3)));
        assert_eq!(min_heap.remove(), Some(('E', 3)));
        assert_eq!(min_heap.remove(), Some(('B', 7)));

        min_heap.insert('A', 1000);
        min_heap.insert('B', 2);
        min_heap.insert('C', 3);
        min_heap.insert('D', 1);
        min_heap.insert('E', -1000);
        assert_eq!(min_heap.remove(), Some(('E', -1000)));

        min_heap.insert('F', -10);
        min_heap.insert('G', 2000);

        assert_eq!(min_heap.remove(), Some(('F', -10)));

        min_heap.insert('H', -2000);
        assert_eq!(min_heap.remove(), Some(('H', -2000)));

        assert_eq!(min_heap.remove(), Some(('D', 1)));
        assert_eq!(min_heap.remove(), Some(('B', 2)));
        assert_eq!(min_heap.remove(), Some(('C', 3)));
        assert_eq!(min_heap.remove(), Some(('A', 1000)));
        assert_eq!(min_heap.remove(), Some(('G', 2000)));

        min_heap.insert('I', -2000);
        min_heap.insert('J', -3000);
        min_heap.insert('K', -4000);
        assert_eq!(min_heap.remove(), Some(('K', -4000)));
        assert_eq!(min_heap.remove(), Some(('J', -3000)));
        assert_eq!(min_heap.remove(), Some(('I', -2000)));
    }

    #[test]
    fn test_max_binary_heap() {
        let mut heap: MaxBinaryHeap<i32> = MaxBinaryHeap::with_capacity(3);
        heap.insert(4);
        heap.insert(5);
        heap.insert(6);
        assert_eq!(heap.capacity, 3);
        heap.insert(7);
        heap.insert(8);
        heap.insert(9);
        assert_eq!(heap.capacity, 6);
        assert_eq!(heap.remove(), Some(9));
        assert_eq!(heap.remove(), Some(8));
        assert_eq!(heap.remove(), Some(7));
        assert_eq!(heap.remove(), Some(6));
        assert_eq!(heap.remove(), Some(5));
        assert_eq!(heap.remove(), Some(4));
        assert_eq!(heap.capacity, 6);
        

        let mut heap: MaxBinaryHeap<i32> = MaxBinaryHeap::with_capacity(10);
        assert_eq!(heap.remove(), None);
        
        heap.insert(1);
        assert_eq!(heap.remove(), Some(1));
        assert_eq!(heap.remove(), None);

        heap.insert(1);
        heap.insert(2);
        heap.insert(3);
        assert_eq!(heap.remove(), Some(3));
        assert_eq!(heap.remove(), Some(2));
        assert_eq!(heap.remove(), Some(1));

        heap.insert(3);
        heap.insert(2);
        heap.insert(1);
        assert_eq!(heap.remove(), Some(3));
        assert_eq!(heap.remove(), Some(2));
        assert_eq!(heap.remove(), Some(1));

        heap.insert(2);
        heap.insert(5);
        heap.insert(1);
        heap.insert(2);
        heap.insert(3);
        assert_eq!(heap.remove(), Some(5));
        assert_eq!(heap.remove(), Some(3));
        assert_eq!(heap.remove(), Some(2));
        assert_eq!(heap.remove(), Some(2));
        assert_eq!(heap.remove(), Some(1));

        heap.insert(3);
        heap.insert(7);
        heap.insert(2);
        heap.insert(1);
        heap.insert(3);
        assert_eq!(heap.remove(), Some(7));
        assert_eq!(heap.remove(), Some(3));
        assert_eq!(heap.remove(), Some(3));
        assert_eq!(heap.remove(), Some(2));
        assert_eq!(heap.remove(), Some(1));

        heap.insert(1000);
        heap.insert(2);
        heap.insert(3);
        heap.insert(1);
        heap.insert(-1000);
        assert_eq!(heap.remove(), Some(1000));

        heap.insert(-10);
        heap.insert(2000);

        assert_eq!(heap.remove(), Some(2000));

        heap.insert(-2000);
        assert_eq!(heap.remove(), Some(3));

        assert_eq!(heap.remove(), Some(2));
        assert_eq!(heap.remove(), Some(1));
        assert_eq!(heap.remove(), Some(-10));
        assert_eq!(heap.remove(), Some(-1000));
        assert_eq!(heap.remove(), Some(-2000));

        heap.insert(-2000);
        heap.insert(-3000);
        heap.insert(-4000);
        assert_eq!(heap.remove(), Some(-2000));
        assert_eq!(heap.remove(), Some(-3000));
        assert_eq!(heap.remove(), Some(-4000));
    }

    #[test]
    fn test_max_binary_heap_2() {
        let mut heap: MaxBinaryHeap2<char, i64> = MaxBinaryHeap2::with_capacity(3);
        heap.insert('A', 4);
        heap.insert('B', 5);
        heap.insert('C', 6);
        assert_eq!(heap.capacity, 3);
        heap.insert('D', 7);
        heap.insert('E', 8);
        heap.insert('F', 9);
        assert_eq!(heap.capacity, 6);
        assert_eq!(heap.remove(), Some(('F', 9)));        
        assert_eq!(heap.remove(), Some(('E', 8)));
        assert_eq!(heap.remove(), Some(('D', 7)));
        assert_eq!(heap.remove(), Some(('C', 6)));
        assert_eq!(heap.remove(), Some(('B', 5)));
        assert_eq!(heap.remove(), Some(('A', 4)));


        let mut heap: MaxBinaryHeap2<char, i64> = MaxBinaryHeap2::with_capacity(10);
        assert_eq!(heap.remove(), None);
        
        heap.insert('A', 1);
        assert_eq!(heap.remove(), Some(('A', 1)));
        assert_eq!(heap.remove(), None);

        heap.insert('A', 1);
        heap.insert('B', 2);
        heap.insert('C', 3);
        assert_eq!(heap.remove(), Some(('C', 3)));
        assert_eq!(heap.remove(), Some(('B', 2)));
        assert_eq!(heap.remove(), Some(('A', 1)));

        heap.insert('A', 3);
        heap.insert('B', 2);
        heap.insert('C', 1);
        assert_eq!(heap.remove(), Some(('A', 3)));
        assert_eq!(heap.remove(), Some(('B', 2)));
        assert_eq!(heap.remove(), Some(('C', 1)));


        heap.insert('A', 2);
        heap.insert('B', 5);
        heap.insert('C', 1);
        heap.insert('D', 2);
        heap.insert('E', 3);
        assert_eq!(heap.remove(), Some(('B', 5)));
        assert_eq!(heap.remove(), Some(('E', 3)));
        assert_eq!(heap.remove(), Some(('D', 2)));
        assert_eq!(heap.remove(), Some(('A', 2)));
        assert_eq!(heap.remove(), Some(('C', 1)));

        heap.insert('A', 3);
        heap.insert('B', 7);
        heap.insert('C', 2);
        heap.insert('D', 1);
        heap.insert('E', 3);
        assert_eq!(heap.remove(), Some(('B', 7)));
        assert_eq!(heap.remove(), Some(('E', 3)));
        assert_eq!(heap.remove(), Some(('A', 3)));
        assert_eq!(heap.remove(), Some(('C', 2)));
        assert_eq!(heap.remove(), Some(('D', 1)));

        heap.insert('A', 1000);
        heap.insert('B', 2);
        heap.insert('C', 3);
        heap.insert('D', 1);
        heap.insert('E', -1000);
        assert_eq!(heap.remove(), Some(('A', 1000)));

        heap.insert('F', -10);
        heap.insert('G', 2000);

        assert_eq!(heap.remove(), Some(('G', 2000)));

        heap.insert('H', -2000);


        assert_eq!(heap.remove(), Some(('C', 3)));
        assert_eq!(heap.remove(), Some(('B', 2)));
        assert_eq!(heap.remove(), Some(('D', 1)));

        assert_eq!(heap.remove(), Some(('F', -10)));
        assert_eq!(heap.remove(), Some(('E', -1000)));

        heap.insert('I', -2000);
        heap.insert('J', -3000);
        heap.insert('K', -4000);
        assert_eq!(heap.remove(), Some(('H', -2000)));
        assert_eq!(heap.remove(), Some(('I', -2000)));
        assert_eq!(heap.remove(), Some(('J', -3000)));
        assert_eq!(heap.remove(), Some(('K', -4000)));
    }
}