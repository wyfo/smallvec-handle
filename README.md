# smallvec-handle

A `Vec`-like implementation backed by a local array. Contrary to other alternative
`Vec` implementation, which use a tagged union to distinguish between local and heap storage,
this one uses a single pointer for both local and heap slice.

Because the pointer may be invalidated when the vector is moved, it must be checked before each
operation on the vector. To avoid needlessly repeating this check, it is done only once, while
retrieving a "handle" to the vector. The handle is then used for every subsequent operations,
hence the crate name.

When the whole data can fit in the local array, it allows saving an allocation, and may have
better cache locality than a regular `Vec`. Also, it should be more performant than using a
tagged union implementation, because it avoids branching at each operation.

## Example

```rust
use smallvec_handle::SmallVec;
let mut vec = SmallVec::<usize, 16>::new();
let vec_handle = vec.handle();
vec_handle.push(0);
vec_handle.push(1);
assert_eq!(vec_handle.as_slice(), [0, 1]);
assert_eq!(vec.as_slice(), [0, 1]);
```