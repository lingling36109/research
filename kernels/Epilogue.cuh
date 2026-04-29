#pragma once
#include "Common.h"
#include <cub/cub.cuh>

namespace somespgemm {

// TODO: change all output to size_t*
template<class T, class src_offset_t=index_t, int THREADS_PER_ROW=32, int CONCURRENT_ROWS=2, int ITER_PER_BLOCK=4>
__global__ __launch_bounds__(THREADS_PER_ROW*CONCURRENT_ROWS, 2048/(THREADS_PER_ROW*CONCURRENT_ROWS)) void copyOutput(
    T* dst, T* src, 
    index_t* offset_dst, src_offset_t* offsets_src, 
    index_t* nums, 
    index_t num_rows,
    index_t* copy_rows=nullptr,
    index_t src_num_cols=0
) {
    constexpr int rows_per_block = CONCURRENT_ROWS * ITER_PER_BLOCK;
    index_t start_row = blockIdx.x * rows_per_block;
    index_t end_row = start_row + rows_per_block < num_rows ? start_row + rows_per_block : num_rows;

    int my_row_id = threadIdx.x / THREADS_PER_ROW;
    int my_element_id = threadIdx.x % THREADS_PER_ROW;

    // use scratchpad memory to store the offset and nums in advance
    __shared__ index_t smem[4 * CONCURRENT_ROWS * ITER_PER_BLOCK];
    index_t* s_offset_dst = smem;
    index_t* s_nums = smem + CONCURRENT_ROWS * ITER_PER_BLOCK;
    src_offset_t* s_offsets_src = (src_offset_t*)(smem + 2 * CONCURRENT_ROWS * ITER_PER_BLOCK);

    // write the s_offset_dst
    for (int i = threadIdx.x; i < end_row - start_row; i += blockDim.x) {
        index_t row_id;


        if (copy_rows != nullptr) {
            row_id = copy_rows[start_row + i];
        } else {
            row_id = start_row + i;
        }
        if (row_id == (index_t)(-1)) {
            s_nums[i] = 0;
            continue;
        }

        s_offset_dst[i] = offset_dst[row_id];
        if (offsets_src != nullptr) {
            s_offsets_src[i] = offsets_src[row_id];
        } else {
            s_offsets_src[i] = src_num_cols * row_id;
        }
        s_nums[i] = nums[row_id];
    }
    __syncthreads();
    
    for (index_t i = start_row + my_row_id; i < end_row; i += CONCURRENT_ROWS) {
        src_offset_t src_offset = s_offsets_src[i - start_row];
        index_t dst_offset = s_offset_dst[i - start_row];
        int num = s_nums[i - start_row];
        int cid = my_element_id;
        while (cid < num) {
            dst[dst_offset + cid] = src[src_offset + cid];
            cid += THREADS_PER_ROW;
        }
    }
}

// Uses dynamic shared memory for radix sorting
template<int items_per_thread, int threads_per_block>
__global__ void sortOutputDyn(
    index_t* col_in, data_t* val_in, index_t* offset_in,
    index_t* col_out, data_t* val_out, index_t* offset_out,
    index_t* my_rows, index_t* num_elements
) {
    using BlockRadixSort = cub::BlockRadixSort<index_t, threads_per_block, items_per_thread, data_t>;
    using BlockLoad = cub::BlockLoad<index_t, threads_per_block, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockLoadVal = cub::BlockLoad<data_t, threads_per_block, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockStore = cub::BlockStore<index_t, threads_per_block, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>;
    using BlockStoreVal = cub::BlockStore<data_t, threads_per_block, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>;

    union shmem_layout{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename BlockLoadVal::TempStorage load_val;
        typename BlockStore::TempStorage store;
        typename BlockStoreVal::TempStorage store_val;
    };

    extern __shared__ __align__(alignof(shmem_layout)) char smem_t[];
    auto& temp_storage = reinterpret_cast<shmem_layout&>(smem_t);

    index_t thread_data[items_per_thread];
    data_t thread_data_val[items_per_thread];

    index_t my_row = my_rows[blockIdx.x];
    if (my_row == (index_t)(-1)) {
        return;
    }
    index_t start = offset_in[my_row];
    index_t my_elements = num_elements[my_row];
    index_t out_start = offset_out[my_row];
    
    if (my_elements <= NUM_HASH_INPLACE_SORT_THRESHOLD) {
        // no sort, just copy
        for (index_t i = threadIdx.x; i < my_elements; i += blockDim.x) {
            col_out[out_start + i] = col_in[start + i];
            val_out[out_start + i] = val_in[start + i];
        }
        return;
    }
        
    BlockLoad(temp_storage.load).Load(col_in + start, thread_data, my_elements, (index_t)(-1));
    __syncthreads();    // This is necessary since we use the same temp_storage for different purposes
    BlockLoadVal(temp_storage.load_val).Load(val_in + start, thread_data_val, my_elements, (index_t)(-1));
    __syncthreads();

    BlockRadixSort(temp_storage.sort).Sort(thread_data, thread_data_val, /*begin_bit=*/0, /*end_bit=*/sizeof(index_t) * 8);
    __syncthreads();

    // store
    BlockStore(temp_storage.store).Store(col_out + out_start, thread_data, my_elements);
    __syncthreads();
    BlockStoreVal(temp_storage.store_val).Store(val_out + out_start, thread_data_val, my_elements);

}

// Fuse col_idx and loc_in_array into a single element
template<int items_per_thread, int threads_per_block, int IDX_BITS>
__global__ __launch_bounds__(threads_per_block, 1024/threads_per_block) void sortOutputFused(
    index_t* col_in, data_t* val_in, index_t* offset_in,
    index_t* col_out, data_t* val_out, index_t* offset_out,
    index_t* my_rows, index_t* num_elements,
    index_t valid_bits = 0
) {
    if (valid_bits == 0) {
        valid_bits = IDX_BITS;
    }

    using BlockRadixSort = cub::BlockRadixSort<index_t, threads_per_block, items_per_thread>;
    using BlockLoad = cub::BlockLoad<index_t, threads_per_block, items_per_thread, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoadVal = cub::BlockLoad<data_t, threads_per_block, items_per_thread, cub::BLOCK_LOAD_STRIPED>;
    using BlockStore = cub::BlockStore<index_t, threads_per_block, items_per_thread, cub::BLOCK_STORE_STRIPED>;
    using BlockExchange = cub::BlockExchange<index_t, threads_per_block, items_per_thread>;

    union shmem_layout{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename BlockLoadVal::TempStorage load_val;
        typename BlockStore::TempStorage store;
        typename BlockExchange::TempStorage exchange;
    };

    extern __shared__ __align__(alignof(shmem_layout)) char smem_t[];
    auto& temp_storage = reinterpret_cast<shmem_layout&>(smem_t);

    index_t thread_data[items_per_thread];

    index_t my_row = my_rows[blockIdx.x];
    if (my_row == (index_t)(-1)) {
        return;
    }
    index_t start = offset_in[my_row];
    index_t my_elements = num_elements[my_row];
    
    if (my_elements <= NUM_HASH_INPLACE_SORT_THRESHOLD) {
        index_t out_start = offset_out[my_row];
        for (index_t i = threadIdx.x; i < my_elements; i += blockDim.x) {
            col_out[out_start + i] = col_in[start + i];
            val_out[out_start + i] = val_in[start + i];
        }
        return;
    }
        
    BlockLoad(temp_storage.load).Load(col_in + start, thread_data, my_elements, (index_t)(-1));
    __syncthreads();    // This is necessary since we use the same temp_storage for different purposes

    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        // the index of the item is stored into the upper bits
        thread_data[i] = ((threads_per_block * i + threadIdx.x) << IDX_BITS) | (thread_data[i]);
    }
    __syncthreads();

    BlockRadixSort(temp_storage.sort).Sort(thread_data, /*begin_bit=*/0, /*end_bit=*/valid_bits);
    __syncthreads();

    // transpose, for direct stripped storing
    BlockExchange(temp_storage.exchange).BlockedToStriped(thread_data);
    __syncthreads();

    // load values into shared memory, in a stripped fashion
    data_t* raw_data = reinterpret_cast<data_t*>(smem_t);
    start = offset_in[my_row];
    for (int i = threadIdx.x; i < my_elements; i += blockDim.x) {
        raw_data[i] = val_in[start + i];
    }
    __syncthreads();

    index_t out_start = offset_out[my_row];
    
    // store the value according to the upper bits, then clean the upper bits
    constexpr unsigned int mask = (1 << IDX_BITS) - 1;

    #pragma unroll 
    for (int i = 0; i < items_per_thread; i++) {
        int dst_idx = i * threads_per_block + threadIdx.x;
        if (dst_idx >= my_elements) {
            continue;
        }

        // source idx is in the upper bits
        int src_idx = thread_data[i] >> IDX_BITS;

        val_out[dst_idx + out_start] = raw_data[src_idx];

        // actual index is in the lower bits
        thread_data[i] &= mask;
    }
    __syncthreads();

    BlockStore(temp_storage.store).Store(col_out + out_start, thread_data, my_elements);
    __syncthreads();
}

}