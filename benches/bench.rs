#![feature(test)]
#![allow(deprecated)]

extern crate test;

use smallvec_handle::SmallVecHandle;
use test::Bencher;

const VEC_SIZE: usize = 16;
const SPILLED_SIZE: usize = 100;

trait Vector<T> {
    type Handle: VectorHandle<T>;
    fn new() -> Self;
    fn handle(&mut self) -> &mut Self::Handle;
    fn from_elem(val: T, n: usize) -> Self;
    fn from_elems(val: &[T]) -> Self;
}

trait VectorHandle<T>: Extend<T> {
    fn push(&mut self, val: T);
    fn pop(&mut self) -> Option<T>;
    fn remove(&mut self, p: usize) -> T;
    fn insert(&mut self, n: usize, val: T);
    fn extend_from_slice(&mut self, other: &[T]);
}

impl<T: Copy> Vector<T> for Vec<T> {
    type Handle = Vec<T>;

    fn new() -> Self {
        Self::with_capacity(VEC_SIZE)
    }

    fn handle(&mut self) -> &mut Self::Handle {
        self
    }

    fn from_elem(val: T, n: usize) -> Self {
        vec![val; n]
    }

    fn from_elems(val: &[T]) -> Self {
        val.to_owned()
    }
}

impl<T: Copy> VectorHandle<T> for Vec<T> {
    fn push(&mut self, val: T) {
        self.push(val)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn remove(&mut self, p: usize) -> T {
        self.remove(p)
    }

    fn insert(&mut self, n: usize, val: T) {
        self.insert(n, val)
    }

    fn extend_from_slice(&mut self, other: &[T]) {
        Vec::extend_from_slice(self, other)
    }
}

impl<T: Copy> Vector<T> for smallvec::SmallVec<T, VEC_SIZE> {
    type Handle = smallvec::SmallVec<T, VEC_SIZE>;

    fn new() -> Self {
        Self::new()
    }

    fn handle(&mut self) -> &mut Self::Handle {
        self
    }

    fn from_elem(val: T, n: usize) -> Self {
        smallvec::smallvec![val; n]
    }

    fn from_elems(val: &[T]) -> Self {
        smallvec::SmallVec::from_slice(val)
    }
}

impl<T: Copy> VectorHandle<T> for smallvec::SmallVec<T, VEC_SIZE> {
    fn push(&mut self, val: T) {
        self.push(val)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn remove(&mut self, p: usize) -> T {
        self.remove(p)
    }

    fn insert(&mut self, n: usize, val: T) {
        self.insert(n, val)
    }

    fn extend_from_slice(&mut self, other: &[T]) {
        self.extend_from_slice(other)
    }
}

impl<T: Copy> Vector<T> for smallvec_handle::SmallVec<T, VEC_SIZE> {
    type Handle = SmallVecHandle<T, VEC_SIZE>;

    fn new() -> Self {
        Self::new()
    }

    fn handle(&mut self) -> &mut Self::Handle {
        self.handle()
    }

    fn from_elem(val: T, n: usize) -> Self {
        let mut vec = Self::new();
        vec.handle().resize(n, val);
        vec
    }

    fn from_elems(val: &[T]) -> Self {
        let mut vec = Self::new();
        vec.handle().extend_from_slice(val);
        vec
    }
}

impl<T: Copy> VectorHandle<T> for SmallVecHandle<T, VEC_SIZE> {
    fn push(&mut self, val: T) {
        self.push(val)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn remove(&mut self, p: usize) -> T {
        self.remove(p)
    }

    fn insert(&mut self, n: usize, val: T) {
        self.insert(n, val)
    }

    fn extend_from_slice(&mut self, other: &[T]) {
        self.extend_from_slice(other)
    }
}

macro_rules! make_benches {
    ($typ:ty { $($b_name:ident => $g_name:ident($($args:expr),*),)* }) => {
        $(
            #[bench]
            fn $b_name(b: &mut Bencher) {
                $g_name::<$typ>($($args,)* b)
            }
        )*
    }
}

make_benches! {
    smallvec_handle::SmallVec<u64, VEC_SIZE> {
        bench_push => gen_push(SPILLED_SIZE as _),
        bench_push_small => gen_push(VEC_SIZE as _),
        bench_insert_push => gen_insert_push(SPILLED_SIZE as _),
        bench_insert_push_small => gen_insert_push(VEC_SIZE as _),
        bench_insert => gen_insert(SPILLED_SIZE as _),
        bench_insert_small => gen_insert(VEC_SIZE as _),
        bench_remove => gen_remove(SPILLED_SIZE as _),
        bench_remove_small => gen_remove(VEC_SIZE as _),
        bench_extend => gen_extend(SPILLED_SIZE as _),
        bench_extend_small => gen_extend(VEC_SIZE as _),
        bench_extend_filtered => gen_extend_filtered(SPILLED_SIZE as _),
        bench_extend_filtered_small => gen_extend_filtered(VEC_SIZE as _),
        bench_from_slice => gen_from_slice(SPILLED_SIZE as _),
        bench_from_slice_small => gen_from_slice(VEC_SIZE as _),
        bench_extend_from_slice => gen_extend_from_slice(SPILLED_SIZE as _),
        bench_extend_from_slice_small => gen_extend_from_slice(VEC_SIZE as _),
        bench_macro_from_elem => gen_from_elem(SPILLED_SIZE as _),
        bench_macro_from_elem_small => gen_from_elem(VEC_SIZE as _),
        bench_pushpop => gen_pushpop(),
    }
}

make_benches! {
    smallvec::SmallVec<u64, VEC_SIZE> {
        smallvec_bench_push => gen_push(SPILLED_SIZE as _),
        smallvec_bench_push_small => gen_push(VEC_SIZE as _),
        smallvec_bench_insert_push => gen_insert_push(SPILLED_SIZE as _),
        smallvec_bench_insert_push_small => gen_insert_push(VEC_SIZE as _),
        smallvec_bench_insert => gen_insert(SPILLED_SIZE as _),
        smallvec_bench_insert_small => gen_insert(VEC_SIZE as _),
        smallvec_bench_remove => gen_remove(SPILLED_SIZE as _),
        smallvec_bench_remove_small => gen_remove(VEC_SIZE as _),
        smallvec_bench_extend => gen_extend(SPILLED_SIZE as _),
        smallvec_bench_extend_small => gen_extend(VEC_SIZE as _),
        smallvec_bench_extend_filtered => gen_extend_filtered(SPILLED_SIZE as _),
        smallvec_bench_extend_filtered_small => gen_extend_filtered(VEC_SIZE as _),
        smallvec_bench_from_slice => gen_from_slice(SPILLED_SIZE as _),
        smallvec_bench_from_slice_small => gen_from_slice(VEC_SIZE as _),
        smallvec_bench_extend_from_slice => gen_extend_from_slice(SPILLED_SIZE as _),
        smallvec_bench_extend_from_slice_small => gen_extend_from_slice(VEC_SIZE as _),
        smallvec_bench_macro_from_elem => gen_from_elem(SPILLED_SIZE as _),
        smallvec_bench_macro_from_elem_small => gen_from_elem(VEC_SIZE as _),
        smallvec_bench_pushpop => gen_pushpop(),
    }
}

make_benches! {
    Vec<u64> {
        vec_bench_push => gen_push(SPILLED_SIZE as _),
        vec_bench_push_small => gen_push(VEC_SIZE as _),
        vec_bench_insert_push => gen_insert_push(SPILLED_SIZE as _),
        vec_bench_insert_push_small => gen_insert_push(VEC_SIZE as _),
        vec_bench_insert => gen_insert(SPILLED_SIZE as _),
        vec_bench_insert_small => gen_insert(VEC_SIZE as _),
        vec_bench_remove => gen_remove(SPILLED_SIZE as _),
        vec_bench_remove_small => gen_remove(VEC_SIZE as _),
        vec_bench_extend => gen_extend(SPILLED_SIZE as _),
        vec_bench_extend_small => gen_extend(VEC_SIZE as _),
        vec_bench_extend_filtered => gen_extend_filtered(SPILLED_SIZE as _),
        vec_bench_extend_filtered_small => gen_extend_filtered(VEC_SIZE as _),
        vec_bench_from_slice => gen_from_slice(SPILLED_SIZE as _),
        vec_bench_from_slice_small => gen_from_slice(VEC_SIZE as _),
        vec_bench_extend_from_slice => gen_extend_from_slice(SPILLED_SIZE as _),
        vec_bench_extend_from_slice_small => gen_extend_from_slice(VEC_SIZE as _),
        vec_bench_macro_from_elem => gen_from_elem(SPILLED_SIZE as _),
        vec_bench_macro_from_elem_small => gen_from_elem(VEC_SIZE as _),
        vec_bench_pushpop => gen_pushpop(),
    }
}

fn gen_push<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    #[inline(never)]
    fn push_noinline<H: VectorHandle<u64>>(handle: &mut H, x: u64) {
        handle.push(x);
    }

    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        for x in 0..n {
            push_noinline(handle, x);
        }
        vec
    });
}

fn gen_insert_push<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    #[inline(never)]
    fn insert_push_noinline<H: VectorHandle<u64>>(handle: &mut H, x: u64) {
        handle.insert(x as usize, x);
    }

    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        for x in 0..n {
            insert_push_noinline(handle, x);
        }
        vec
    });
}

fn gen_insert<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    #[inline(never)]
    fn insert_noinline<H: VectorHandle<u64>>(handle: &mut H, p: usize, x: u64) {
        handle.insert(p, x)
    }

    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        // Always insert at position 0 so that we are subject to shifts of
        // many different lengths.
        handle.push(0);
        for x in 0..n {
            insert_noinline(handle, 0, x);
        }
        vec
    });
}

fn gen_remove<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    #[inline(never)]
    fn remove_noinline<H: VectorHandle<u64>>(handle: &mut H, p: usize) -> u64 {
        handle.remove(p)
    }

    b.iter(|| {
        let mut vec = V::from_elem(0, n as _);
        let handle = V::handle(&mut vec);

        for _ in 0..n {
            remove_noinline(handle, 0);
        }
    });
}

fn gen_extend<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        handle.extend(0..n);
        vec
    });
}

fn gen_extend_filtered<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        handle.extend((0..n).filter(|i| i % 2 == 0));
        vec
    });
}

fn gen_from_slice<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    let v: Vec<u64> = (0..n).collect();
    b.iter(|| {
        let vec = V::from_elems(&v);
        vec
    });
}

fn gen_extend_from_slice<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    let v: Vec<u64> = (0..n).collect();
    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        handle.extend_from_slice(&v);
        vec
    });
}

fn gen_pushpop<V: Vector<u64>>(b: &mut Bencher) {
    #[inline(never)]
    fn pushpop_noinline<H: VectorHandle<u64>>(handle: &mut H, x: u64) -> Option<u64> {
        handle.push(x);
        handle.pop()
    }

    b.iter(|| {
        let mut vec = V::new();
        let handle = V::handle(&mut vec);
        for x in 0..SPILLED_SIZE as _ {
            pushpop_noinline(handle, x);
        }
        vec
    });
}

fn gen_from_elem<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    b.iter(|| {
        let vec = V::from_elem(42, n);
        vec
    });
}
