pub struct MinBinaryHeap<T> {
    data: Vec<T>,
    capacity: usize,
    size: usize,
}

impl<T> MinBinaryHeap<T>
where
    T: Default + Copy + PartialOrd,
{
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data: Vec<T> = Vec::with_capacity(0);
        data.resize(capacity, T::default());
        Self {
            data,
            capacity,
            size: 0,
        }
    }
    pub fn push(&mut self, value: T) {
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

    pub fn pop(&mut self) -> Option<T> {
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
        while (current > left && left_position < size) || (current > right && right_position < size)
        {
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
        if pos > 0 {
            (pos - 1) / 2
        } else {
            0
        }
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
        pos >= self.size / 2
    }

    #[inline(always)]
    fn swap(&mut self, first: usize, second: usize) {
        let buffer: T = self.data[first];
        self.data[first] = self.data[second];
        self.data[second] = buffer;
    }
}

pub struct MinBinaryHeap2<T, W> {
    data: Vec<(T, W)>,
    capacity: usize,
    size: usize,
}

impl<T, W> MinBinaryHeap2<T, W>
where
    T: Default + Copy,
    W: Default + Copy + PartialOrd,
{
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data: Vec<(T, W)> = Vec::with_capacity(0);
        data.resize(capacity, (T::default(), W::default()));
        Self {
            data,
            capacity,
            size: 0,
        }
    }
    pub fn push(&mut self, value: T, weight: W) {
        if self.size >= self.capacity {
            self.data
                .resize(self.capacity * 2 + 1, (T::default(), W::default()));
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

    pub fn pop(&mut self) -> Option<(T, W)> {
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
        while (current.1 > left.1 && left_position < size)
            || (current.1 > right.1 && right_position < size)
        {
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
        if pos > 0 {
            (pos - 1) / 2
        } else {
            0
        }
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
        pos >= self.size / 2
    }

    #[inline(always)]
    fn swap(&mut self, first: usize, second: usize) {
        let buffer: (T, W) = self.data[first];
        self.data[first] = self.data[second];
        self.data[second] = buffer;
    }
}

pub struct MaxBinaryHeap<T> {
    data: Vec<T>,
    capacity: usize,
    size: usize,
}

impl<T> MaxBinaryHeap<T>
where
    T: Default + Copy + PartialOrd,
{
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data: Vec<T> = Vec::with_capacity(0);
        data.resize(capacity, T::default());
        Self {
            data,
            capacity,
            size: 0,
        }
    }
    pub fn push(&mut self, value: T) {
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

    pub fn pop(&mut self) -> Option<T> {
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
        while (current < left && left_position < size) || (current < right && right_position < size)
        {
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
        if pos > 0 {
            (pos - 1) / 2
        } else {
            0
        }
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
        pos >= self.size / 2
    }

    #[inline(always)]
    fn swap(&mut self, first: usize, second: usize) {
        let buffer: T = self.data[first];
        self.data[first] = self.data[second];
        self.data[second] = buffer;
    }
}

pub struct MaxBinaryHeap2<T, W> {
    data: Vec<(T, W)>,
    capacity: usize,
    size: usize,
}

impl<T, W> MaxBinaryHeap2<T, W>
where
    T: Default + Copy,
    W: Default + Copy + PartialOrd,
{
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data: Vec<(T, W)> = Vec::with_capacity(0);
        data.resize(capacity, (T::default(), W::default()));
        Self {
            data: vec![(T::default(), W::default()); capacity],
            capacity,
            size: 0,
        }
    }
    pub fn push(&mut self, value: T, weight: W) {
        if self.size >= self.capacity {
            self.data
                .resize(self.capacity * 2 + 1, (T::default(), W::default()));
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

    pub fn pop(&mut self) -> Option<(T, W)> {
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
        while (current.1 < left.1 && left_position < size)
            || (current.1 < right.1 && right_position < size)
        {
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
        if pos > 0 {
            (pos - 1) / 2
        } else {
            0
        }
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
        pos >= self.size / 2
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
        min_heap.push(4);
        min_heap.push(5);
        min_heap.push(6);
        assert_eq!(min_heap.capacity, 3);
        min_heap.push(7);
        min_heap.push(8);
        min_heap.push(9);
        assert_eq!(min_heap.capacity, 6);
        assert_eq!(min_heap.pop(), Some(4));
        assert_eq!(min_heap.pop(), Some(5));
        assert_eq!(min_heap.pop(), Some(6));
        assert_eq!(min_heap.pop(), Some(7));
        assert_eq!(min_heap.pop(), Some(8));
        assert_eq!(min_heap.pop(), Some(9));

        let mut min_heap: MinBinaryHeap<i32> = MinBinaryHeap::with_capacity(10);
        assert_eq!(min_heap.pop(), None);

        min_heap.push(1);
        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), None);

        min_heap.push(1);
        min_heap.push(2);
        min_heap.push(3);
        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(2));
        assert_eq!(min_heap.pop(), Some(3));

        min_heap.push(3);
        min_heap.push(2);
        min_heap.push(1);
        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(2));
        assert_eq!(min_heap.pop(), Some(3));

        min_heap.push(2);
        min_heap.push(5);
        min_heap.push(1);
        min_heap.push(2);
        min_heap.push(3);
        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(2));
        assert_eq!(min_heap.pop(), Some(2));
        assert_eq!(min_heap.pop(), Some(3));
        assert_eq!(min_heap.pop(), Some(5));

        min_heap.push(3);
        min_heap.push(7);
        min_heap.push(2);
        min_heap.push(1);
        min_heap.push(3);
        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(2));
        assert_eq!(min_heap.pop(), Some(3));
        assert_eq!(min_heap.pop(), Some(3));
        assert_eq!(min_heap.pop(), Some(7));

        min_heap.push(1000);
        min_heap.push(2);
        min_heap.push(3);
        min_heap.push(1);
        min_heap.push(-1000);
        assert_eq!(min_heap.pop(), Some(-1000));

        min_heap.push(-10);
        min_heap.push(2000);

        assert_eq!(min_heap.pop(), Some(-10));

        min_heap.push(-2000);
        assert_eq!(min_heap.pop(), Some(-2000));

        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(2));
        assert_eq!(min_heap.pop(), Some(3));
        assert_eq!(min_heap.pop(), Some(1000));
        assert_eq!(min_heap.pop(), Some(2000));

        min_heap.push(-2000);
        min_heap.push(-3000);
        min_heap.push(-4000);
        assert_eq!(min_heap.pop(), Some(-4000));
        assert_eq!(min_heap.pop(), Some(-3000));
        assert_eq!(min_heap.pop(), Some(-2000));
    }

    #[test]
    fn test_min_binary_heap_2() {
        let mut min_heap: MinBinaryHeap2<char, i64> = MinBinaryHeap2::with_capacity(3);
        min_heap.push('A', 4);
        min_heap.push('B', 5);
        min_heap.push('C', 6);
        assert_eq!(min_heap.capacity, 3);
        min_heap.push('D', 7);
        min_heap.push('E', 8);
        min_heap.push('F', 9);
        assert_eq!(min_heap.capacity, 6);
        assert_eq!(min_heap.pop(), Some(('A', 4)));
        assert_eq!(min_heap.pop(), Some(('B', 5)));
        assert_eq!(min_heap.pop(), Some(('C', 6)));
        assert_eq!(min_heap.pop(), Some(('D', 7)));
        assert_eq!(min_heap.pop(), Some(('E', 8)));
        assert_eq!(min_heap.pop(), Some(('F', 9)));

        let mut min_heap: MinBinaryHeap2<char, i64> = MinBinaryHeap2::with_capacity(10);
        assert_eq!(min_heap.pop(), None);

        min_heap.push('A', 1);
        assert_eq!(min_heap.pop(), Some(('A', 1)));
        assert_eq!(min_heap.pop(), None);

        min_heap.push('A', 1);
        min_heap.push('B', 2);
        min_heap.push('C', 3);
        assert_eq!(min_heap.pop(), Some(('A', 1)));
        assert_eq!(min_heap.pop(), Some(('B', 2)));
        assert_eq!(min_heap.pop(), Some(('C', 3)));

        min_heap.push('A', 3);
        min_heap.push('B', 2);
        min_heap.push('C', 1);
        assert_eq!(min_heap.pop(), Some(('C', 1)));
        assert_eq!(min_heap.pop(), Some(('B', 2)));
        assert_eq!(min_heap.pop(), Some(('A', 3)));

        min_heap.push('A', 2);
        min_heap.push('B', 5);
        min_heap.push('C', 1);
        min_heap.push('D', 2);
        min_heap.push('E', 3);
        assert_eq!(min_heap.pop(), Some(('C', 1)));
        assert_eq!(min_heap.pop(), Some(('A', 2)));
        assert_eq!(min_heap.pop(), Some(('D', 2)));
        assert_eq!(min_heap.pop(), Some(('E', 3)));
        assert_eq!(min_heap.pop(), Some(('B', 5)));

        min_heap.push('A', 3);
        min_heap.push('B', 7);
        min_heap.push('C', 2);
        min_heap.push('D', 1);
        min_heap.push('E', 3);
        assert_eq!(min_heap.pop(), Some(('D', 1)));
        assert_eq!(min_heap.pop(), Some(('C', 2)));
        assert_eq!(min_heap.pop(), Some(('A', 3)));
        assert_eq!(min_heap.pop(), Some(('E', 3)));
        assert_eq!(min_heap.pop(), Some(('B', 7)));

        min_heap.push('A', 1000);
        min_heap.push('B', 2);
        min_heap.push('C', 3);
        min_heap.push('D', 1);
        min_heap.push('E', -1000);
        assert_eq!(min_heap.pop(), Some(('E', -1000)));

        min_heap.push('F', -10);
        min_heap.push('G', 2000);

        assert_eq!(min_heap.pop(), Some(('F', -10)));

        min_heap.push('H', -2000);
        assert_eq!(min_heap.pop(), Some(('H', -2000)));

        assert_eq!(min_heap.pop(), Some(('D', 1)));
        assert_eq!(min_heap.pop(), Some(('B', 2)));
        assert_eq!(min_heap.pop(), Some(('C', 3)));
        assert_eq!(min_heap.pop(), Some(('A', 1000)));
        assert_eq!(min_heap.pop(), Some(('G', 2000)));

        min_heap.push('I', -2000);
        min_heap.push('J', -3000);
        min_heap.push('K', -4000);
        assert_eq!(min_heap.pop(), Some(('K', -4000)));
        assert_eq!(min_heap.pop(), Some(('J', -3000)));
        assert_eq!(min_heap.pop(), Some(('I', -2000)));
    }

    #[test]
    fn test_max_binary_heap() {
        let mut heap: MaxBinaryHeap<i32> = MaxBinaryHeap::with_capacity(3);
        heap.push(4);
        heap.push(5);
        heap.push(6);
        assert_eq!(heap.capacity, 3);
        heap.push(7);
        heap.push(8);
        heap.push(9);
        assert_eq!(heap.capacity, 6);
        assert_eq!(heap.pop(), Some(9));
        assert_eq!(heap.pop(), Some(8));
        assert_eq!(heap.pop(), Some(7));
        assert_eq!(heap.pop(), Some(6));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.capacity, 6);

        let mut heap: MaxBinaryHeap<i32> = MaxBinaryHeap::with_capacity(10);
        assert_eq!(heap.pop(), None);

        heap.push(1);
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);

        heap.push(1);
        heap.push(2);
        heap.push(3);
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));

        heap.push(3);
        heap.push(2);
        heap.push(1);
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));

        heap.push(2);
        heap.push(5);
        heap.push(1);
        heap.push(2);
        heap.push(3);
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));

        heap.push(3);
        heap.push(7);
        heap.push(2);
        heap.push(1);
        heap.push(3);
        assert_eq!(heap.pop(), Some(7));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));

        heap.push(1000);
        heap.push(2);
        heap.push(3);
        heap.push(1);
        heap.push(-1000);
        assert_eq!(heap.pop(), Some(1000));

        heap.push(-10);
        heap.push(2000);

        assert_eq!(heap.pop(), Some(2000));

        heap.push(-2000);
        assert_eq!(heap.pop(), Some(3));

        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(-10));
        assert_eq!(heap.pop(), Some(-1000));
        assert_eq!(heap.pop(), Some(-2000));

        heap.push(-2000);
        heap.push(-3000);
        heap.push(-4000);
        assert_eq!(heap.pop(), Some(-2000));
        assert_eq!(heap.pop(), Some(-3000));
        assert_eq!(heap.pop(), Some(-4000));
    }

    #[test]
    fn test_max_binary_heap_2() {
        let mut heap: MaxBinaryHeap2<char, i64> = MaxBinaryHeap2::with_capacity(3);
        heap.push('A', 4);
        heap.push('B', 5);
        heap.push('C', 6);
        assert_eq!(heap.capacity, 3);
        heap.push('D', 7);
        heap.push('E', 8);
        heap.push('F', 9);
        assert_eq!(heap.capacity, 6);
        assert_eq!(heap.pop(), Some(('F', 9)));
        assert_eq!(heap.pop(), Some(('E', 8)));
        assert_eq!(heap.pop(), Some(('D', 7)));
        assert_eq!(heap.pop(), Some(('C', 6)));
        assert_eq!(heap.pop(), Some(('B', 5)));
        assert_eq!(heap.pop(), Some(('A', 4)));

        let mut heap: MaxBinaryHeap2<char, i64> = MaxBinaryHeap2::with_capacity(10);
        assert_eq!(heap.pop(), None);

        heap.push('A', 1);
        assert_eq!(heap.pop(), Some(('A', 1)));
        assert_eq!(heap.pop(), None);

        heap.push('A', 1);
        heap.push('B', 2);
        heap.push('C', 3);
        assert_eq!(heap.pop(), Some(('C', 3)));
        assert_eq!(heap.pop(), Some(('B', 2)));
        assert_eq!(heap.pop(), Some(('A', 1)));

        heap.push('A', 3);
        heap.push('B', 2);
        heap.push('C', 1);
        assert_eq!(heap.pop(), Some(('A', 3)));
        assert_eq!(heap.pop(), Some(('B', 2)));
        assert_eq!(heap.pop(), Some(('C', 1)));

        heap.push('A', 2);
        heap.push('B', 5);
        heap.push('C', 1);
        heap.push('D', 2);
        heap.push('E', 3);
        assert_eq!(heap.pop(), Some(('B', 5)));
        assert_eq!(heap.pop(), Some(('E', 3)));
        assert_eq!(heap.pop(), Some(('D', 2)));
        assert_eq!(heap.pop(), Some(('A', 2)));
        assert_eq!(heap.pop(), Some(('C', 1)));

        heap.push('A', 3);
        heap.push('B', 7);
        heap.push('C', 2);
        heap.push('D', 1);
        heap.push('E', 3);
        assert_eq!(heap.pop(), Some(('B', 7)));
        assert_eq!(heap.pop(), Some(('E', 3)));
        assert_eq!(heap.pop(), Some(('A', 3)));
        assert_eq!(heap.pop(), Some(('C', 2)));
        assert_eq!(heap.pop(), Some(('D', 1)));

        heap.push('A', 1000);
        heap.push('B', 2);
        heap.push('C', 3);
        heap.push('D', 1);
        heap.push('E', -1000);
        assert_eq!(heap.pop(), Some(('A', 1000)));

        heap.push('F', -10);
        heap.push('G', 2000);

        assert_eq!(heap.pop(), Some(('G', 2000)));

        heap.push('H', -2000);

        assert_eq!(heap.pop(), Some(('C', 3)));
        assert_eq!(heap.pop(), Some(('B', 2)));
        assert_eq!(heap.pop(), Some(('D', 1)));

        assert_eq!(heap.pop(), Some(('F', -10)));
        assert_eq!(heap.pop(), Some(('E', -1000)));

        heap.push('I', -2000);
        heap.push('J', -3000);
        heap.push('K', -4000);
        assert_eq!(heap.pop(), Some(('H', -2000)));
        assert_eq!(heap.pop(), Some(('I', -2000)));
        assert_eq!(heap.pop(), Some(('J', -3000)));
        assert_eq!(heap.pop(), Some(('K', -4000)));
    }
}
