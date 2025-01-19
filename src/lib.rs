#![forbid(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![forbid(clippy::dbg_macro)]
#![forbid(clippy::semicolon_if_nothing_returned)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
//! A [`Vec`]-like container that can store a small number of elements inline.
//!
//! Contrary to other alternative small vector implementation, which use a tagged union to
//! distinguish between local and heap storage, this one uses a single pointer for both local and
//! heap slice.
//!
//! Because the pointer may be invalidated when the vector is moved, it must be checked before each
//! operation on the vector. To avoid needlessly repeating this check, it is done only once, while
//! retrieving a "handle" to the vector. The handle is then used for every subsequent operations,
//! hence the crate name.
//!
//! When the whole data can fit in the local array, it allows saving an allocation, and may have
//! better cache locality than a regular `Vec`. Also, it can be more performant than using a
//! tagged union implementation, because it avoids branching at each operation.
//!
//! # Examples
//!
//! ```
//! # use smallvec_handle::SmallVec;
//! let mut vec = SmallVec::<usize, 16>::new();
//! let mut vec_handle = vec.handle();
//! vec_handle.push(0);
//! vec_handle.push(1);
//! assert_eq!(vec_handle, [0, 1]);
//! assert_eq!(vec.as_slice(), [0, 1]);
//! ```

extern crate alloc;
extern crate std;

use alloc::{collections::TryReserveError, vec::Vec};
use core::{
    borrow::{Borrow, BorrowMut},
    cmp::{max, Ordering},
    fmt,
    hash::{Hash, Hasher},
    hint,
    iter::FusedIterator,
    mem,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Bound, Deref, DerefMut, RangeBounds},
    ptr,
    ptr::NonNull,
    slice,
};

/// A [`Vec`]-like container with a local storage.
///
/// It is mutated through a [`SmallVecHandle`] returned by [`handle`](SmallVec::handle) method.
/// See [crate documentation](crate).
///
/// `SmallVec` doesn't derive `Deref`/`DerefMut` as the as
/// [`as_slice`](SmallVec::as_slice)/[`as_mut_slice`](SmallVec::as_mut_slice) operations are not
/// trivial.
pub struct SmallVec<T, const N: usize> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    local: [MaybeUninit<T>; N],
}

// SAFETY: the data referenced by the inner pointer is unaliased
unsafe impl<T: Send, const N: usize> Send for SmallVec<T, N> {}
// SAFETY: the data referenced by the inner pointer is unaliased
unsafe impl<T: Sync, const N: usize> Sync for SmallVec<T, N> {}

impl<T, const N: usize> SmallVec<T, N> {
    const ASSERT: () = assert!(N > 0 && core::mem::size_of::<T>() > 0);
    /// Construct a new, empty `SmallVec<T, N>`.
    ///
    /// The vector is initialized with a capacity N.
    #[inline]
    pub const fn new() -> Self {
        #[allow(path_statements)]
        Self::ASSERT;
        // SAFETY: An uninitialized `[MaybeUninit<_>; N]` is valid (previous implementation of
        // `MaybeUninit::uninit_array`)
        // `[const { MaybeUninit::uninit() }; N]` syntax requires 1.79
        let local = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: N,
            local,
        }
    }

    #[inline(always)]
    const fn is_local(&self) -> bool {
        self.cap == N
    }

    /// # Safety
    ///
    /// `SmallVec` must have been allocated from a vector, see [`Self::grow`].
    #[inline(always)]
    unsafe fn get_vec(&self) -> Vec<T> {
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap) }
    }

    /// Returns a [`SmallVecHandle`] reference.
    #[inline]
    pub fn handle(&mut self) -> SmallVecHandle<T, N> {
        if self.is_local() {
            self.ptr = NonNull::new(self.local.as_mut_ptr().cast()).unwrap();
        }
        SmallVecHandle(self)
    }

    /// Returns the total number of elements the vector can hold without
    /// reallocating.
    ///
    /// See [Vec::capacity]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// See [Vec::len]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// See [Vec::is_empty]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the vector's buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn't allocate.
    ///
    /// See [Vec::as_ptr]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        if self.is_local() {
            self.local.as_ptr().cast()
        } else {
            self.ptr.as_ptr()
        }
    }

    /// Returns a raw mutable pointer to the vector's buffer, or a dangling
    /// raw pointer valid for zero sized reads if the vector didn't allocate.
    ///
    /// See [Vec::as_mut_ptr]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        if self.is_local() {
            self.local.as_mut_ptr().cast()
        } else {
            self.ptr.as_ptr()
        }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// See [Vec::set_len]
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`](Self::capacity).
    /// - The elements at `old_len..new_len` must be initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        self.len = new_len;
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// See [Vec::as_slice]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// See [Vec::as_mut_slice]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// See [Vec::spare_capacity_mut]
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        // SAFETY: copied from stdlib
        unsafe {
            slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.len) as *mut MaybeUninit<T>,
                self.cap - self.len,
            )
        }
    }

    /// Converts the vector into [`Box<[T]>`](Box).
    ///
    /// See [Vec::into_boxed_slice]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        if self.is_local() {
            let this = ManuallyDrop::new(self);
            let mut vec = Vec::with_capacity(this.len);
            unsafe {
                ptr::copy_nonoverlapping(this.local.as_ptr().cast(), vec.as_mut_ptr(), this.len);
            }
            vec.into_boxed_slice()
        } else {
            self.into_vec().into()
        }
    }

    /// Converts the vector into [`Vec`].
    pub fn into_vec(self) -> Vec<T> {
        let this = ManuallyDrop::new(self);
        if this.is_local() {
            let mut vec = Vec::with_capacity(N);
            let local_ptr = this.local.as_ptr().cast();
            unsafe {
                ptr::copy_nonoverlapping(local_ptr, vec.as_mut_ptr(), this.len);
            }
            vec
        } else {
            unsafe { this.get_vec() }
        }
    }
}

impl<T, const N: usize> Drop for SmallVec<T, N> {
    fn drop(&mut self) {
        let guard = DeallocGuard(self);
        let slice = ptr::slice_from_raw_parts_mut(guard.0.as_mut_ptr(), guard.0.len());
        unsafe { ptr::drop_in_place(slice) };
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        fn init_from_vec<T, const N: usize>(this: &mut SmallVec<T, N>, mut vec: Vec<T>) {
            this.ptr = NonNull::new(vec.as_mut_ptr()).unwrap();
            this.len = vec.len();
            this.cap = vec.capacity();
        }

        let mut this = Self::new();
        let mut iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > N {
            #[cold]
            fn from_iter_impl<T, const N: usize>(
                this: &mut SmallVec<T, N>,
                lower: usize,
                iter: impl IntoIterator<Item = T>,
            ) {
                // Do not use `Vec::from_iter` to ensure the capacity is greater than N
                // (handling the case where `lower` is wrong)
                let mut vec = Vec::with_capacity(lower);
                vec.extend(iter);
                init_from_vec(this, vec);
            }
            from_iter_impl(&mut this, lower, iter);
            return this;
        }
        for i in 0..N {
            match iter.next() {
                Some(item) => this.local[i] = MaybeUninit::new(item),
                None => return this,
            }
            this.len = i;
        }
        if let Some(item) = iter.next() {
            #[cold]
            fn from_iter_impl<T, const N: usize>(
                this: &mut SmallVec<T, N>,
                item: T,
                iter: impl IntoIterator<Item = T>,
            ) {
                let mut vec = Vec::<T>::with_capacity(2 * N);
                unsafe {
                    ptr::copy_nonoverlapping(this.local.as_ptr().cast(), vec.as_mut_ptr(), N);
                    ptr::write(vec.as_mut_ptr().add(N + 1), item);
                    vec.set_len(N + 1);
                }
                vec.extend(iter);
                init_from_vec(this, vec);
            }
            from_iter_impl(&mut this, item, iter);
        }
        this
    }
}

impl<T, const N: usize> IntoIterator for SmallVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            vec: ManuallyDrop::new(self),
            cur: 0,
        }
    }
}

/// An iterator that moves out of a vector.
///
/// See [`alloc::vec::IntoIter`]
pub struct IntoIter<T, const N: usize> {
    vec: ManuallyDrop<SmallVec<T, N>>,
    cur: usize,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.vec.len() {
            return None;
        }
        let item = unsafe { self.vec.as_ptr().add(self.cur).read() };
        self.cur += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.vec.len() - self.cur;
        (size, Some(size))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cur == self.vec.len() {
            return None;
        }
        self.vec.len -= 1;
        let item = unsafe { self.vec.as_ptr().add(self.vec.len()).read() };
        Some(item)
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        let guard = DeallocGuard(&mut self.vec);
        let ptr = unsafe { guard.0.as_mut_ptr().add(self.cur) };
        let slice = ptr::slice_from_raw_parts_mut(ptr, guard.0.len() - self.cur);
        unsafe { ptr::drop_in_place(slice) };
    }
}

struct DeallocGuard<'a, T, const N: usize>(&'a mut SmallVec<T, N>);

impl<T, const N: usize> Drop for DeallocGuard<'_, T, N> {
    #[inline(always)]
    fn drop(&mut self) {
        let handle = &mut self.0;
        if !handle.is_local() {
            let ptr = handle.ptr.as_ptr().cast::<MaybeUninit<T>>();
            drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, handle.cap)) });
        }
    }
}

/// A "handle" to mutate a [`SmallVec`] instance.
///
/// It provides a similar API to `Vec`.
pub struct SmallVecHandle<'a, T, const N: usize>(&'a mut SmallVec<T, N>);

impl<T, const N: usize> SmallVecHandle<'_, T, N> {
    /// Returns the total number of elements the vector can hold without
    /// reallocating.
    ///
    /// See [Vec::capacity]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.0.cap
    }

    #[inline(always)]
    fn need_grow(&self, additional: usize) -> bool {
        self.0.len + additional > self.0.cap
    }

    /// # Safety
    /// `need_grow(additional)` must have returned `true`
    #[inline(always)]
    unsafe fn grow(
        &mut self,
        mut additional: usize,
        try_: bool,
        exact: bool,
    ) -> Result<(), TryReserveError> {
        let this = &mut *self.0;
        let mut vec = ManuallyDrop::new(if this.is_local() {
            if exact {
                additional += this.len;
            } else {
                additional = max(this.len + additional, 2 * N);
            }
            Vec::<T>::new()
        } else {
            if !exact {
                additional = max(additional, 2 * this.cap - this.len);
            }
            // SAFETY: cap != N means the vector has been recreated from a vec
            // (see branch above)
            unsafe { this.get_vec() }
        });
        if additional <= vec.capacity() - vec.len() {
            // SAFETY: `need_grow` has returned `true`
            unsafe { hint::unreachable_unchecked() };
        }
        if try_ {
            vec.try_reserve_exact(additional)?;
        } else {
            vec.reserve_exact(additional);
        }
        if this.cap == N {
            // SAFETY: src and dst are valid and non-overlapping
            unsafe { ptr::copy_nonoverlapping(this.as_ptr(), vec.as_mut_ptr(), this.len) };
        }
        this.ptr = NonNull::new(vec.as_mut_ptr()).unwrap();
        this.cap = vec.capacity();
        Ok(())
    }

    /// # Safety
    /// `need_grow(1)` must have returned `true`
    #[inline(never)]
    unsafe fn grow_one(&mut self) {
        // SAFETY: `need_grow(1)` has returned `true`
        unsafe { self.grow(1, false, false).unwrap() };
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`.
    ///
    /// See [Vec::reserve]
    pub fn reserve(&mut self, additional: usize) {
        if self.need_grow(additional) {
            #[cold]
            fn grow<T, const N: usize>(vec: &mut SmallVecHandle<T, N>, additional: usize) {
                // SAFETY: `need_grow` has returned `true`
                unsafe { vec.grow(additional, false, false).unwrap() };
            }
            grow(self, additional);
        }
    }

    /// Reserves the minimum capacity for at least `additional` more elements to
    /// be inserted in the given `Vec<T>`.
    ///
    /// See [Vec::reserve_exact]
    pub fn reserve_exact(&mut self, additional: usize) {
        if self.need_grow(additional) {
            // SAFETY: `need_grow` has returned `true`
            unsafe { self.grow(additional, false, true).unwrap() };
        }
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`.
    ///
    /// See [Vec::try_reserve]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        if self.need_grow(additional) {
            // SAFETY: `need_grow` has returned `true`
            unsafe { self.grow(additional, true, false)? };
        }
        Ok(())
    }

    /// Tries to reserve the minimum capacity for at least `additional`
    /// elements to be inserted in the given `Vec<T>`.
    ///
    /// See [Vec::try_reserve_exact]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        if self.need_grow(additional) {
            // SAFETY: `need_grow` has returned `true`
            unsafe { self.grow(additional, true, true)? };
        }
        Ok(())
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// See [Vec::truncate]
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        let this = &mut *self.0;
        // SAFETY: copied from stdlib
        // This is safe because:
        //
        // * the slice passed to `drop_in_place` is valid; the `len > self.len`
        //   case avoids creating an invalid slice, and
        // * the `len` of the vector is shrunk before calling `drop_in_place`,
        //   such that no value will be dropped twice in case `drop_in_place`
        //   were to panic once (if it panics twice, the program aborts).
        unsafe {
            // Note: It's intentional that this is `>` and not `>=`.
            //       Changing it to `>=` has negative performance
            //       implications in some cases. See #78884 for more.
            if len > this.len {
                return;
            }
            let remaining_len = this.len - len;
            let s = ptr::slice_from_raw_parts_mut(this.as_mut_ptr().add(len), remaining_len);
            this.len = len;
            ptr::drop_in_place(s);
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// See [Vec::as_slice]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: `slice::from_raw_parts` requires pointee is a contiguous, aligned buffer of size
        // `len` containing properly-initialized `T`s. Data must not be mutated for the returned
        // lifetime. Further, `len * mem::size_of::<T>` <= `ISIZE::MAX`, and allocation does not
        // "wrap" through overflowing memory addresses.
        //
        // * Vec API guarantees that self.buf:
        //      * contains only properly-initialized items within 0..len
        //      * is aligned, contiguous, and valid for `len` reads
        //      * obeys size and address-wrapping constraints
        //
        // * We only construct `&mut` references to `self.buf` through `&mut self` methods; borrow-
        //   check ensures that it is not possible to mutably alias `self.buf` within the
        //   returned lifetime.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.0.len) }
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// See [Vec::as_mut_slice]
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `slice::from_raw_parts_mut` requires pointee is a contiguous, aligned buffer of
        // size `len` containing properly-initialized `T`s. Data must not be accessed through any
        // other pointer for the returned lifetime. Further, `len * mem::size_of::<T>` <=
        // `ISIZE::MAX` and allocation does not "wrap" through overflowing memory addresses.
        //
        // * Vec API guarantees that self.buf:
        //      * contains only properly-initialized items within 0..len
        //      * is aligned, contiguous, and valid for `len` reads
        //      * obeys size and address-wrapping constraints
        //
        // * We only construct references to `self.buf` through `&self` and `&mut self` methods;
        //   borrow-check ensures that it is not possible to construct a reference to `self.buf`
        //   within the returned lifetime.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.0.len) }
    }

    /// Returns a raw pointer to the vector's buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn't allocate.
    ///
    /// See [Vec::as_ptr]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.0.ptr.as_ptr().cast_const()
    }

    /// Returns a raw mutable pointer to the vector's buffer, or a dangling
    /// raw pointer valid for zero sized reads if the vector didn't allocate.
    ///
    /// See [Vec::as_mut_ptr]
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.ptr.as_ptr()
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// See [Vec::set_len]
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`](Self::capacity).
    /// - The elements at `old_len..new_len` must be initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        // SAFETY: same function contract
        unsafe { self.0.set_len(new_len) }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// See [Vec::swap_remove]
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        #[cold]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("swap_remove index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }
        // SAFETY: copied from stdlib
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check above succeeds there must be a last element (which
            // can be self[index] itself).
            let value = ptr::read(self.as_ptr().add(index));
            let base_ptr = self.as_mut_ptr();
            ptr::copy(base_ptr.add(len - 1), base_ptr.add(index), 1);
            self.set_len(len - 1);
            value
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// See [Vec::insert]
    pub fn insert(&mut self, index: usize, element: T) {
        #[cold]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        let len = self.len();
        if index > len {
            assert_failed(index, len);
        }

        // space for the new element
        if self.need_grow(1) {
            // SAFETY: `need_grow` has returned `true`
            unsafe { self.grow_one() };
        }

        // SAFETY: copied from stdlib
        unsafe {
            // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().add(index);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    ptr::copy(p, p.add(1), len - index);
                }
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.set_len(len + 1);
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// See [Vec::remove]
    pub fn remove(&mut self, index: usize) -> T {
        #[cold]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }
        // SAFETY: copied from stdlib
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.add(1), ptr, len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// See [Vec::retain]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// See [Vec::retain_mut]
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let original_len = self.len();

        if original_len == 0 {
            // Empty case: explicit return allows better optimization, vs letting compiler infer it
            return;
        }

        // SAFETY: copied from stdlib
        // Avoid double drop if the drop guard is not executed,
        // since we may make some holes during the process.
        unsafe { self.set_len(0) };

        // Vec: [Kept, Kept, Hole, Hole, Hole, Hole, Unchecked, Unchecked]
        //      |<-              processed len   ->| ^- next to check
        //                  |<-  deleted cnt     ->|
        //      |<-              original_len                          ->|
        // Kept: Elements which predicate returns true on.
        // Hole: Moved or dropped element slot.
        // Unchecked: Unchecked valid elements.
        //
        // This drop guard will be invoked when predicate or `drop` of element panicked.
        // It shifts unchecked elements to cover holes and `set_len` to the correct length.
        // In cases when predicate and `drop` never panick, it will be optimized out.
        struct BackshiftOnDrop<'a, T, const N: usize> {
            v: &'a mut SmallVec<T, N>,
            processed_len: usize,
            deleted_cnt: usize,
            original_len: usize,
        }

        impl<T, const N: usize> Drop for BackshiftOnDrop<'_, T, N> {
            fn drop(&mut self) {
                if self.deleted_cnt > 0 {
                    // SAFETY: Trailing unchecked items must be valid since we never touch them.
                    unsafe {
                        ptr::copy(
                            self.v.as_ptr().add(self.processed_len),
                            self.v
                                .as_mut_ptr()
                                .add(self.processed_len - self.deleted_cnt),
                            self.original_len - self.processed_len,
                        );
                    }
                }
                // SAFETY: After filling holes, all items are in contiguous memory.
                unsafe {
                    self.v.set_len(self.original_len - self.deleted_cnt);
                }
            }
        }

        let mut g = BackshiftOnDrop {
            v: &mut *self.0,
            processed_len: 0,
            deleted_cnt: 0,
            original_len,
        };

        fn process_loop<F, T, const N: usize, const DELETED: bool>(
            original_len: usize,
            f: &mut F,
            g: &mut BackshiftOnDrop<'_, T, N>,
        ) where
            F: FnMut(&mut T) -> bool,
        {
            while g.processed_len != original_len {
                // SAFETY: Unchecked element must be valid.
                let cur = unsafe { &mut *g.v.as_mut_ptr().add(g.processed_len) };
                if !f(cur) {
                    // Advance early to avoid double drop if `drop_in_place` panicked.
                    g.processed_len += 1;
                    g.deleted_cnt += 1;
                    // SAFETY: We never touch this element again after dropped.
                    unsafe { ptr::drop_in_place(cur) };
                    // We already advanced the counter.
                    if DELETED {
                        continue;
                    } else {
                        break;
                    }
                }
                if DELETED {
                    // SAFETY: `deleted_cnt` > 0, so the hole slot must not overlap with current element.
                    // We use copy for move, and never touch this element again.
                    unsafe {
                        let hole_slot = g.v.as_mut_ptr().add(g.processed_len - g.deleted_cnt);
                        ptr::copy_nonoverlapping(cur, hole_slot, 1);
                    }
                }
                g.processed_len += 1;
            }
        }

        // Stage 1: Nothing was deleted.
        process_loop::<F, T, N, false>(original_len, &mut f, &mut g);

        // Stage 2: Some elements were deleted.
        process_loop::<F, T, N, true>(original_len, &mut f, &mut g);

        // All item are processed. This can be optimized to `set_len` by LLVM.
        drop(g);
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same
    /// key.
    ///
    /// See [Vec::dedup_by_key]
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality
    /// relation.
    ///
    /// See [Vec::dedup_by]
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let len = self.len();
        if len <= 1 {
            return;
        }

        // Check if we ever want to remove anything.
        // This allows to use copy_non_overlapping in next cycle.
        // And avoids any memory writes if we don't need to remove anything.
        let mut first_duplicate_idx: usize = 1;
        let start = self.as_mut_ptr();
        while first_duplicate_idx != len {
            // SAFETY: copied from stdlib
            let found_duplicate = unsafe {
                // SAFETY: first_duplicate always in range [1..len)
                // Note that we start iteration from 1 so we never overflow.
                let prev = start.add(first_duplicate_idx.wrapping_sub(1));
                let current = start.add(first_duplicate_idx);
                // We explicitly say in docs that references are reversed.
                same_bucket(&mut *current, &mut *prev)
            };
            if found_duplicate {
                break;
            }
            first_duplicate_idx += 1;
        }
        // Don't need to remove anything.
        // We cannot get bigger than len.
        if first_duplicate_idx == len {
            return;
        }

        /* INVARIANT: vec.len() > read > write > write-1 >= 0 */
        struct FillGapOnDrop<'a, T, const N: usize> {
            /* Offset of the element we want to check if it is duplicate */
            read: usize,

            /* Offset of the place where we want to place the non-duplicate
             * when we find it. */
            write: usize,

            /* The Vec that would need correction if `same_bucket` panicked */
            vec: &'a mut SmallVec<T, N>,
        }

        impl<T, const N: usize> Drop for FillGapOnDrop<'_, T, N> {
            fn drop(&mut self) {
                /* This code gets executed when `same_bucket` panics */

                /* SAFETY: invariant guarantees that `read - write`
                 * and `len - read` never overflow and that the copy is always
                 * in-bounds. */
                unsafe {
                    let ptr = self.vec.as_mut_ptr();
                    let len = self.vec.len();

                    /* How many items were left when `same_bucket` panicked.
                     * Basically vec[read..].len() */
                    let items_left = len.wrapping_sub(self.read);

                    /* Pointer to first item in vec[write..write+items_left] slice */
                    let dropped_ptr = ptr.add(self.write);
                    /* Pointer to first item in vec[read..] slice */
                    let valid_ptr = ptr.add(self.read);

                    /* Copy `vec[read..]` to `vec[write..write+items_left]`.
                     * The slices can overlap, so `copy_nonoverlapping` cannot be used */
                    ptr::copy(valid_ptr, dropped_ptr, items_left);

                    /* How many items have been already dropped
                     * Basically vec[read..write].len() */
                    let dropped = self.read.wrapping_sub(self.write);

                    self.vec.set_len(len - dropped);
                }
            }
        }

        /* Drop items while going through Vec, it should be more efficient than
         * doing slice partition_dedup + truncate */

        // Construct gap first and then drop item to avoid memory corruption if `T::drop` panics.
        let mut gap = FillGapOnDrop {
            read: first_duplicate_idx + 1,
            write: first_duplicate_idx,
            vec: &mut *self.0,
        };
        // SAFETY: copied from stdlib
        unsafe {
            // SAFETY: we checked that first_duplicate_idx in bounds before.
            // If drop panics, `gap` would remove this item without drop.
            ptr::drop_in_place(start.add(first_duplicate_idx));
        }

        /* SAFETY: Because of the invariant, read_ptr, prev_ptr and write_ptr
         * are always in-bounds and read_ptr never aliases prev_ptr */
        unsafe {
            while gap.read < len {
                let read_ptr = start.add(gap.read);
                let prev_ptr = start.add(gap.write.wrapping_sub(1));

                // We explicitly say in docs that references are reversed.
                let found_duplicate = same_bucket(&mut *read_ptr, &mut *prev_ptr);
                if found_duplicate {
                    // Increase `gap.read` now since the drop may panic.
                    gap.read += 1;
                    /* We have found duplicate, drop it in-place */
                    ptr::drop_in_place(read_ptr);
                } else {
                    let write_ptr = start.add(gap.write);

                    /* read_ptr cannot be equal to write_ptr because at this point
                     * we guaranteed to skip at least one element (before loop starts).
                     */
                    ptr::copy_nonoverlapping(read_ptr, write_ptr, 1);

                    /* We have filled that place, so go further */
                    gap.write += 1;
                    gap.read += 1;
                }
            }

            /* Technically we could let `gap` clean up with its Drop, but
             * when `same_bucket` is guaranteed to not panic, this bloats a little
             * the codegen, so we just do it manually */
            gap.vec.set_len(gap.write);
            mem::forget(gap);
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// See [Vec::push]
    #[inline]
    pub fn push(&mut self, value: T) {
        // Inform codegen that the length does not change across grow_one().
        let len = self.0.len;
        // This will panic or abort if we would allocate > isize::MAX bytes
        // or if the length increment would overflow for zero-sized types.
        if self.need_grow(1) {
            // SAFETY: `need_grow` has returned `true`
            unsafe { self.grow_one() };
        }
        // SAFETY: copied from stdlib
        unsafe {
            let end = self.as_mut_ptr().add(len);
            ptr::write(end, value);
            self.0.len = len + 1;
        }
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// See [Vec::pop]
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.0.len == 0 {
            None
        } else {
            // SAFETY: copied from stdlib
            unsafe {
                self.0.len -= 1;
                if self.0.len >= self.capacity() {
                    hint::unreachable_unchecked();
                }
                Some(ptr::read(self.as_ptr().add(self.len())))
            }
        }
    }

    /// Removes the specified range from the vector in bulk, returning all
    /// removed elements as an iterator.
    ///
    /// See [Vec::drain]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, N>
    where
        R: RangeBounds<usize>,
    {
        // Memory safety
        //
        // When the Drain is first created, it shortens the length of
        // the source vector to make sure no uninitialized or moved-from elements
        // are accessible at all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        // When finished, remaining tail of the vec is copied back to cover
        // the hole, and the vector length is restored to the new length.
        //
        let len = self.len();

        let start = match range.start_bound() {
            Bound::Included(&start1) => start1,
            Bound::Excluded(start1) => start1
                .checked_add(1)
                .unwrap_or_else(|| panic!("attempted to index slice from after maximum usize")),
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(end1) => end1
                .checked_add(1)
                .unwrap_or_else(|| panic!("attempted to index slice up to maximum usize")),
            Bound::Excluded(&end1) => end1,
            Bound::Unbounded => len,
        };

        if start > end {
            panic!("slice index start is larger than end");
        }
        if end > len {
            panic!("slice end index is out of range for slice");
        }

        // SAFETY: copied from stdlib
        unsafe {
            // set self.vec length's to start, to be safe in case Drain is leaked
            self.set_len(start);
            let range_slice = slice::from_raw_parts(self.as_ptr().add(start), end - start);
            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: NonNull::from(&mut *self.0),
            }
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// See [Vec::clear]
    #[inline]
    pub fn clear(&mut self) {
        let elems: *mut [T] = self.as_mut_slice();

        // SAFETY:
        // - `elems` comes directly from `as_mut_slice` and is therefore valid.
        // - Setting `self.len` before calling `drop_in_place` means that,
        //   if an element's `Drop` impl panics, the vector's `Drop` impl will
        //   do nothing (leaking the rest of the elements) instead of dropping
        //   some twice.
        unsafe {
            self.0.len = 0;
            ptr::drop_in_place(elems);
        }
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// See [Vec::len]
    #[inline]
    pub const fn len(&self) -> usize {
        self.0.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// See [Vec::is_empty]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// See [Vec::resize_with]
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        let len = self.len();
        if new_len > len {
            self.reserve(new_len - len);
            let mut cur = self.as_mut_ptr();
            unsafe {
                let end = cur.add(new_len - len);
                while cur != end.sub(1) {
                    ptr::write(cur, f());
                    cur = cur.add(1);
                }
                ptr::write(cur, f());
                self.set_len(new_len);
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// See [Vec::spare_capacity_mut]
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        // SAFETY: copied from stdlib
        unsafe {
            slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.0.len) as *mut MaybeUninit<T>,
                self.0.cap - self.0.len,
            )
        }
    }

    /// Reborrows a handle
    #[inline]
    pub fn reborrow(&mut self) -> SmallVecHandle<'_, T, N> {
        SmallVecHandle(&mut *self.0)
    }
}

impl<T: Clone, const N: usize> SmallVecHandle<'_, T, N> {
    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// See [Vec::resize]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.reserve(new_len - len);
            let mut cur = self.as_mut_ptr();
            unsafe {
                let end = cur.add(new_len - len);
                while cur != end.sub(1) {
                    ptr::write(cur, value.clone());
                    cur = cur.add(1);
                }
                ptr::write(cur, value);
                self.set_len(new_len);
            }
        } else {
            self.truncate(new_len);
        }
    }
    /// Clones and appends all elements in a slice to the vector.
    ///
    /// See [Vec::extend_from_slice]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        let count = other.len();
        self.reserve(count);
        let len = self.len();
        // SAFETY: copied from stdlib
        unsafe { ptr::copy_nonoverlapping(other.as_ptr(), self.as_mut_ptr().add(len), count) };
        self.0.len += count;
    }
}

impl<T: PartialEq, const N: usize> SmallVecHandle<'_, T, N> {
    /// Removes consecutive repeated elements in the vector according to the
    /// [`PartialEq`] trait implementation.
    ///
    /// See [Vec::dedup]
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b);
    }
}

impl<T, const N: usize> Deref for SmallVecHandle<'_, T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for SmallVecHandle<'_, T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Hash, const N: usize> Hash for SmallVecHandle<'_, T, N> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state);
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SmallVecHandle<'_, T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut SmallVecHandle<'_, T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const N: usize> Extend<T> for SmallVecHandle<'_, T, N> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for item in iter {
            self.push(item);
        }
    }
}

impl<'a, T: Copy + 'a, const N: usize> Extend<&'a T> for SmallVecHandle<'_, T, N> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for &item in iter {
            self.push(item);
        }
    }
}

impl<T, U, const N1: usize, const N2: usize> PartialEq<SmallVecHandle<'_, U, N2>>
    for SmallVecHandle<'_, T, N1>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &SmallVecHandle<U, N2>) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize> PartialEq<[U]> for SmallVecHandle<'_, T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize> PartialEq<&[U]> for SmallVecHandle<'_, T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<[U; M]> for SmallVecHandle<'_, T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; M]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<&[U; M]> for SmallVecHandle<'_, T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U; M]) -> bool {
        self[..] == other[..]
    }
}

impl<T, const N: usize> PartialOrd for SmallVecHandle<'_, T, N>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Eq, const N: usize> Eq for SmallVecHandle<'_, T, N> {}

impl<T: Ord, const N: usize> Ord for SmallVecHandle<'_, T, N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallVecHandle<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T, const N: usize> AsRef<[T]> for SmallVecHandle<'_, T, N> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> AsMut<[T]> for SmallVecHandle<'_, T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, const N: usize> Borrow<[T]> for SmallVecHandle<'_, T, N> {
    fn borrow(&self) -> &[T] {
        &self[..]
    }
}

impl<T, const N: usize> BorrowMut<[T]> for SmallVecHandle<'_, T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

/// A draining iterator for `Vec<T>`.
///
/// See [alloc::vec::Drain]
pub struct Drain<'a, T, const N: usize> {
    tail_start: usize,
    tail_len: usize,
    iter: slice::Iter<'a, T>,
    vec: NonNull<SmallVec<T, N>>,
}

impl<T, const N: usize> Drain<'_, T, N> {
    /// Returns the remaining items of this iterator as a slice.
    ///
    /// See [Drain::as_slice](alloc::vec::Drain::as_slice)
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.iter.as_slice()
    }
}

impl<T, const N: usize> AsRef<[T]> for Drain<'_, T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

// SAFETY: copied from stdlib
unsafe impl<T: Send, const N: usize> Send for Drain<'_, T, N> {}
// SAFETY: copied from stdlib
unsafe impl<T: Sync, const N: usize> Sync for Drain<'_, T, N> {}

impl<T, const N: usize> Iterator for Drain<'_, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter
            .next()
            // SAFETY: copied from stdlib
            .map(|elt| unsafe { ptr::read(elt as *const _) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, const N: usize> DoubleEndedIterator for Drain<'_, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter
            .next_back()
            // SAFETY: copied from stdlib
            .map(|elt| unsafe { ptr::read(elt as *const _) })
    }
}

impl<T, const N: usize> ExactSizeIterator for Drain<'_, T, N> {}
impl<T, const N: usize> FusedIterator for Drain<'_, T, N> {}

impl<T, const N: usize> Drop for Drain<'_, T, N> {
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        struct DropGuard<'r, 'a, T, const N: usize>(&'r mut Drain<'a, T, N>);

        impl<T, const N: usize> Drop for DropGuard<'_, '_, T, N> {
            fn drop(&mut self) {
                if self.0.tail_len > 0 {
                    // SAFETY: copied from stdlib
                    unsafe {
                        let source_vec = self.0.vec.as_mut();
                        // memmove back untouched tail, update to new length
                        let start = source_vec.len();
                        let tail = self.0.tail_start;
                        if tail != start {
                            let src = source_vec.as_ptr().add(tail);
                            let dst = source_vec.as_mut_ptr().add(start);
                            ptr::copy(src, dst, self.0.tail_len);
                        }
                        source_vec.set_len(start + self.0.tail_len);
                    }
                }
            }
        }

        let iter = mem::take(&mut self.iter);
        let drop_len = iter.len();

        let mut vec = self.vec;

        // ensure elements are moved back into their appropriate places, even when drop_in_place panics
        let _guard = DropGuard(self);

        if drop_len == 0 {
            return;
        }

        // as_slice() must only be called when iter.len() is > 0 because
        // it also gets touched by vec::Splice which may turn it into a dangling pointer
        // which would make it and the vec pointer point to different allocations which would
        // lead to invalid pointer arithmetic below.
        let drop_ptr = iter.as_slice().as_ptr();

        // SAFETY: copied from stdlib
        unsafe {
            // drop_ptr comes from a slice::Iter which only gives us a &[T] but for drop_in_place
            // a pointer with mutable provenance is necessary. Therefore we must reconstruct
            // it from the original vec but also avoid creating a &mut to the front since that could
            // invalidate raw pointers to it which some unsafe code might rely on.
            let vec_ptr = vec.as_mut().as_mut_ptr();
            let drop_offset = usize::try_from(drop_ptr.offset_from(vec_ptr)).unwrap_unchecked();
            let to_drop = ptr::slice_from_raw_parts_mut(vec_ptr.add(drop_offset), drop_len);
            ptr::drop_in_place(to_drop);
        }
    }
}
