from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out[o] = fn(
                a_storage[index_to_position(a_index, a_strides)],
                b_storage[index_to_position(b_index, b_strides)],
            )

        # if i < out_size:
        #     to_index(i, out_shape, out_index)
        #     o = index_to_position(out_index, out_strides)
        #     broadcast_index(out_index, out_shape, a_shape, a_index)
        #     j = index_to_position(a_index, a_strides)
        #     broadcast_index(out_index, out_shape, b_shape, b_index)
        #     k = index_to_position(b_index, b_strides)
        #     out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    b = cuda.blockIdx.x

    # TODO: Implement for Task 3.3.

    # cache[pos] = a[i] if (i + 1) <= size else 0.0
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()
    r = 2
    if i < size:
        while i % r == 0 and r <= BLOCK_DIM:
            cache[pos] = cache[pos] + cache[pos + r // 2]
            cuda.syncthreads()
            # r = min(BLOCK_DIM + 1, r * 2)
            r *= 2
    if pos == 0:
        out[b] = cache[pos]
        # print('out at {} is {}'.format(b, cache[pos]))

    # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # i = cuda.blockIDx.x * cuda.blockDim.x + cuda.threadIdx.x
    # pos = cuda.threadIdx.x

    # if i < size:
    #     val = float(a[i])
    #     cache[pos] = val
    #     cuda.syncthreads()
    # else:
    #     cache[pos] = 0.0

    # if i < size:
    #     for j in [1, 2, 4, 8, 16]:
    #         if pos % (j * 2) == 0:
    #             cache[pos] += cache[pos + j]
    #             cuda.syncthreads()
    #         if pos == 0:
    #             out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    # print('out: {}'.format(out))
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        to_index(out_pos, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        out_index[reduce_dim] = pos
        a_pos = index_to_position(out_index, a_strides)
        cache[pos] = a_storage[a_pos] if pos < a_shape[reduce_dim] else reduce_value
        cuda.syncthreads()
        r = 2
        while pos % r == 0 and r <= BLOCK_DIM:
            cache[pos] = fn(cache[pos], cache[pos + r // 2])
            cuda.syncthreads()
            r *= 2
        if pos == 0:
            out[out_pos] = cache[pos]

        # cache[pos] = reduce_value

        # if out_pos < out_size:
        #     to_index(out_pos, out_shape, out_index)
        #     o = index_to_position(out_index, out_strides)
        #     # Now we know where we are going. (haven't used thread)

        #     out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
        #     if out_index[reduce_dim] < a_shape[reduce_dim]:
        #         in_a = index_to_position(out_index, a_strides)
        #         cache[pos] = a_storage[in_a]
        #         cuda.syncthread()
        #         x = 0
        #         while 2 ** x < BLOCK_DIM:
        #             j = 2 ** x
        #             if pos % (j * 2) == 0:
        #                 cache[pos] = fn(cache[pos], cache[pos + j])
        #                 cuda.syncthreads()
        #             x += 1
        #     if pos == 0:
        #         out[o] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    pos_i = cuda.threadIdx.x
    pos_j = cuda.threadIdx.y
    i = cuda.blockIdx.x * cuda.blockDim.x + pos_i
    j = cuda.blockIdx.y * cuda.blockDim.y + pos_j
    if i < size and j < size:
        a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
        b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
        t = 0
        for s in range(0, size, BLOCK_DIM):
            # print('ashared posi: {}, ashared posj: {}, apos: {}'.format(pos_i, pos_j,i * size + s + j))
            # print('bshared posi: {}, bshared posj: {}, bpos: {}'.format(pos_i, pos_j,(s + i) * size + j))
            a_shared[pos_i, pos_j] = a[i * size + s + j]
            b_shared[pos_i, pos_j] = b[(s + i) * size + j]
            cuda.syncthreads()

            for k in range(BLOCK_DIM):
                t += a_shared[pos_i, k] * b_shared[k, pos_j]
        out[i * size + j] = t

    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # i = cuda.threadIdx.x
    # j = cuda.threadIdx.you

    # if i >= size or j >= size:
    #     return

    # a_shared[i, j] = a[size * i + j]
    # b_shared[i, j] = b[size * i + j]
    # cuda.syncthreads()

    # accum = 0.0
    # for k in range(size):
    #     accum += a_shared[i, k] * b_shared[k, j]

    # out[size * i + j] = accum


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    t = 0.0
    for blk in range(0, a_shape[2], BLOCK_DIM):
        n = blk + pj
        if i< a_shape[1] and n < a_shape[2]:
            a_shared[pi,pj] = a_storage[
                (a_strides[1] * i) + (a_strides[2] * n) + (batch * a_batch_stride)
            ]
        n = blk + pi
        if n < b_shape[1] and j < b_shape[2]:
            b_shared[pi,pj] = b_storage[
                (b_strides[1] * n) + (b_strides[2] * j ) + (batch * b_batch_stride)
            ]
        cuda.syncthreads()
        for k in range(BLOCK_DIM):
            if k + blk < a_shape[2]:
                t += a_shared[pi,k] * b_shared[k,pj]

    if i < out_shape[1] and j < out_shape[2]:
        o_i = (out_strides[1] * i) + (out_strides[2] * j) + (out_strides[0] * batch)
        out[o_i] = t

    # accum = 0.0
    # for k_start in range(0, a_shape[2], BLOCK_DIM):
    #     k = k_start + pj
    #     if i < a_shape[1] and k < a_shape[2]:
    #         a_shared[pi, pj] = a_storage[
    #             a_batch_stride * batch + a_strides[1] * i + a_strides[2] * j
    #         ]
    #     k = k_start + pi
    #     if j < b_shape[2] and k < b_shape[1]:
    #         b_shared[pi, pj] = b_storage[
    #             b_batch_stride * batch + b_strides[1] * k + b_strides[2] * j
    #         ]
    #     cuda.syncthreads()

    #     for k in range(BLOCK_DIM):
    #         if (k_start + k) < a_shape[2]:
    #             accum += a_shared[pi, k] * b_shared[k, pj]

    # if i < out_shape[1] and j < out_shape[2]:
    #     out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = accum

    # 1) Move across shared dimension by block dim.
    # com_size = a_shape[-2]
    # com_size = a_shape[-1]
    # k_size = a_shape[-1]

    # if i< a_shape[-2] and j< a_shape[-1]:
    #     # for all active threads, set total to zero.
    #     t = 0.0

    #     # for k in range(0,com_size + BLOCK_DIM,BLOCK_DIM):

    #     # for each block, initialize shared memory to zero
    #     cuda.syncthreads()

    #     # for each block, move a and b into shared storage

    #     for k in range(BLOCK_DIM):
    #         if i < a_shape[-2] and (k+pj) < com_size:
    #             a_val = a_storage[batch * a_batch_stride + a_strides[1] * i + a_strides[2] * (k + pj)]
    #             a_shared[pi,pj] = a_val
    #         if (k+pi) < com_size and j< b_shape[-1]:
    #             b_val = b_storage[batch * b_batch_stride + b_strides[1] * (k + pi) + b_strides[2] * j]
    #             b_shared[pi,pj] = b_val
    #         cuda.syncthreads()

    #     for n in range(BLOCK_DIM):
    #         t = t + a_shared[pi,n] * b_shared[n,pj]

    #     out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
    #     out[out_pos] = t

    # I, J, K = out_shape[1], out_shape[2], a_shape[-1]

    # if i < I and j < J:
    #     a_shared[pi,pj] = 0.0
    #     b_shared[pi,pj] = 0.0
    # cuda.syncthreads()

    # for s in range(0, K, BLOCK_DIM):
    #     acc = 0.0
    #     a_k = s + pj
    #     if i < I and a_k < K:
    #         a_shared[pi, pj] = a_storage[
    #             batch * a_batch_stride + i * a_strides[1] + a_k * a_strides[2]
    #         ]
    #     b_k = s + pi
    #     if b_k < K and j < J:
    #         b_shared[pi, pj] = b_storage[
    #             batch * b_batch_stride + b_k * b_strides[1] + j * b_strides[2]
    #         ]
    #     cuda.syncthreads()

    #     for k in range(BLOCK_DIM):
    #         if s + k < K:
    #             acc += a_shared[pi, s] * b_shared[s, pj]
    #     # cuda.syncthreads()

    # if i < I and j < J:
    #     out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
