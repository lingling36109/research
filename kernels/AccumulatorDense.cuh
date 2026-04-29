#pragma once
#include "AccumulatorCommon.cuh"
#include "DeviceCommon.cuh"
#include "Common.h"
#include <cuda_runtime.h>

namespace somespgemm {


// Scan non-zero elements in a dense accumulator
__device__ __forceinline__ index_t scanDenseAccumulator(uint32_t* accumulator, index_t accumulator_size) {
    // do per thread scan
    index_t local_count = 0;
    for (int i = threadIdx.x; i < accumulator_size / (sizeof(uint32_t) * 8); i += blockDim.x) {
        uint32_t val = accumulator[i];
        local_count += __popc(val);
    }

    // do warp level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    __syncthreads();

    // use the first shared memory of accumulator as a temporary storage
    index_t* shared_counts = (index_t*)(accumulator);
    if (threadIdx.x % WARP_SIZE == 0) {
        shared_counts[threadIdx.x / WARP_SIZE] = local_count;
    }
    __syncthreads();

    // first warp does final reduction
    if (threadIdx.x < WARP_SIZE) {
        index_t warp_count = (threadIdx.x < blockDim.x  / WARP_SIZE) ? shared_counts[threadIdx.x] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warp_count += __shfl_down_sync(0xffffffff, warp_count, offset);
        }
        return warp_count;
    }
    return 0;
}

template<int LOG_NTHR, bool QUERY_BITMAP=true>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void denseSymbolicKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *num_products, 
    index_t *max_elements_b, index_t *start_element_b, index_t *end_element_b,
    index_t *my_row_ids,
    index_t *out_nnz_per_row,
    index_t accumulator_size) {
    // accumulator size is in bits
    
    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    uint32_t* accumulator = (uint32_t*)&dynamicShared[0];

    index_t my_row;
    if (my_row_ids == nullptr) {
        my_row = blockIdx.x;
    } else {
        my_row = my_row_ids[blockIdx.x];
    }

    index_t start = start_element_b[my_row];

    index_t valid_size = ((end_element_b[my_row] - start + 31) / 32 + 2) * 32; // in bits
    if (accumulator_size > valid_size) accumulator_size = valid_size; 

    // init accumulator
    for (index_t i = tid; i < accumulator_size / (sizeof(uint32_t) * 8); i += blockDim.x) {
        (accumulator)[i] = 0;
    }
    __syncthreads();
    
    index_t start_element_a = matA_row_offsets[my_row];
    index_t end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];

    // load balance is still the same
    index_t log_nthr_per_a_elem = localLoadBalance(
        end_element_a - start_element_a,
        num_products[my_row],
        max_elements_b[my_row],
        2, LOG_NTHR 
    );

    index_t my_group_id = threadIdx.x >> log_nthr_per_a_elem;
    index_t num_groups = blockDim.x >> log_nthr_per_a_elem;
    index_t my_id_in_group = threadIdx.x & ((1 << log_nthr_per_a_elem) - 1);
    index_t group_size = 1 << log_nthr_per_a_elem;

    for (index_t element_a = start_element_a + my_group_id;
         element_a < end_element_a;
         element_a += num_groups) {

        index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
        index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            index_t b_col = matB_col_ind[b_element] - start;

            // insert into accumulator
            index_t element_pos = b_col / 32;
            index_t bit_pos = b_col % 32;
            if (QUERY_BITMAP) {
                if ((accumulator[element_pos] & (1 << bit_pos)) == 0) {
                    atomicOr(&accumulator[element_pos], (1 << bit_pos));
                }
            } else {
                atomicOr(&accumulator[element_pos], (1 << bit_pos));
            }
        }
        //__syncwarp();
    }

    __syncthreads();
    
    index_t result = scanDenseAccumulator(accumulator, accumulator_size);
    if (tid == 0) {
        out_nnz_per_row[my_row] = result;
    }
}


// when check overflow is enabled, the output position would be compared to the 
// output_size_limit; if exceed, the overflow is marked by returning -1
// all threads are expected to return the same value
template<bool CHECK_OVERFLOW=false>
__device__ __forceinline__ index_t denseCompact(
    data_t* accumulator, index_t accumulator_size, index_t start_col_offset,
    uint32_t* bitmap, 
    uint32_t* prefix_sums,
    index_t* idx_out, data_t* val_out,
    index_t output_size_limit=(index_t)(-1)
) {
    index_t previous_iters_sum = 0;
    if (threadIdx.x == 0) prefix_sums[blockDim.x / 32] = 0;
    if (threadIdx.x == 0) prefix_sums[blockDim.x / 32 + 1] = 0; // initialize the last element
    if (threadIdx.x < blockDim.x / 32) {
        prefix_sums[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int total_iters = (accumulator_size + blockDim.x -1 ) / blockDim.x;
    for (int it = 0; it < total_iters; it ++ ) {

        // step 0: init previous_iters_sum
        __syncthreads();
        previous_iters_sum = prefix_sums[blockDim.x / 32 + 1]; 

        if (CHECK_OVERFLOW) {
            if (previous_iters_sum == (index_t)(-1)) {
                break;
            }
        }

        // step 1: compute prefix sums for each chunk
        uint32_t val = 0;

        if (threadIdx.x < blockDim.x / 32) {
            int current_element = (it * blockDim.x / 32) + threadIdx.x;

            if (it * blockDim.x + threadIdx.x * 32 < accumulator_size) {
                val = __popc(bitmap[current_element]);
            }
        }

        if (threadIdx.x < WARP_SIZE) {
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                uint32_t temp = __shfl_up_sync(0xffffffff, val, offset);
                if ((threadIdx.x) >= offset) {
                    val += temp;
                }
            }

            if (threadIdx.x < blockDim.x / 32) {
                prefix_sums[threadIdx.x + 1] = val;
            }
        }

        __syncthreads();

        // step 2: get my current prefix sum for all previous chunks
        index_t previous_chunks_sum = prefix_sums[threadIdx.x / 32];
        index_t my_current_element = it * blockDim.x + threadIdx.x;

        // step a: update previous_iters_sum for next iteration
        if (threadIdx.x == blockDim.x - 1) {
            prefix_sums[blockDim.x / 32 + 1] = previous_iters_sum + prefix_sums[blockDim.x / 32];
        }

        if (my_current_element >= accumulator_size) {
            break;
        }

        // step 3: get my prefix sum in the current chunk
        uint32_t my_bitmap = bitmap[my_current_element / 32];
        index_t my_bit_pos = my_current_element % 32;
        uint32_t mask = (1 << my_bit_pos) - 1;
        uint32_t masked_bitmap = my_bitmap & mask;
        index_t my_prefix_sum = __popc(masked_bitmap);

        // step 4: write output if necessary
        if ((my_bitmap & (1 << my_bit_pos)) != 0) {
            index_t output_pos = previous_iters_sum + previous_chunks_sum + my_prefix_sum; // there should be not -1 ?
            if (CHECK_OVERFLOW) {
                if (output_pos >= output_size_limit) {
                    // mark overflow
                    prefix_sums[blockDim.x / 32 + 1] = (index_t)(-1);
                    continue;
                }
            }
            idx_out[output_pos] = start_col_offset + my_current_element;
            val_out[output_pos] = accumulator[my_current_element];
        }
    }

    __syncthreads();
    return prefix_sums[blockDim.x / 32 + 1];
}


template<int LOG_NTHR, bool CHECK_OVERFLOW=false, bool QUERY_BITMAP=true>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void denseNumericKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *output_row_offsets, index_t *out_cols, data_t *out_vals,
    index_t *num_products, 
    index_t *max_elements_b, index_t *start_element_b, index_t *end_element_b,
    index_t *inout_my_row_ids,
    index_t accumulator_size,
    index_t *out_nnz_per_row=nullptr,
    index_t *out_num_overflow_rows=nullptr,
    index_t *out_overflow_row_ids=nullptr,
    index_t *out_overflow_buffer_allocated = nullptr
    ) {
    // accumulator size is number of elements
    
    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    data_t* accumulator = (data_t*)&dynamicShared[0];
    uint32_t* bitmap = (uint32_t*)&dynamicShared[accumulator_size * sizeof(data_t)];
    uint32_t* prefix_sums = (uint32_t*)&dynamicShared[accumulator_size * sizeof(data_t) + ((accumulator_size + 31) / 32) * sizeof(uint32_t)];

    index_t my_row;
    if (inout_my_row_ids == nullptr) {
        my_row = blockIdx.x;
    } else {
        my_row = inout_my_row_ids[blockIdx.x];
    }

    index_t start = start_element_b[my_row];
    index_t valid_size = end_element_b[my_row] - start + 1;
    if (accumulator_size > valid_size) accumulator_size = valid_size; 

    // init accumulator
    for (index_t i = tid; i < accumulator_size; i += blockDim.x) {
        accumulator[i] = 0;
    }

    for (index_t i = tid; i < (accumulator_size + 31) / 32; i += blockDim.x) {
        bitmap[i] = 0;
    }
    __syncthreads();
    
    index_t start_element_a = matA_row_offsets[my_row];
    index_t end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];

    index_t log_nthr_per_a_elem = localLoadBalance(
        end_element_a - start_element_a,
        num_products[my_row],
        max_elements_b[my_row],
        2, LOG_NTHR 
    );

    index_t my_group_id = threadIdx.x >> log_nthr_per_a_elem;
    index_t num_groups = blockDim.x >> log_nthr_per_a_elem;
    index_t my_id_in_group = threadIdx.x & ((1 << log_nthr_per_a_elem) - 1);
    index_t group_size = 1 << log_nthr_per_a_elem;

    for (index_t element_a = start_element_a + my_group_id;
         element_a < end_element_a;
         element_a += num_groups) {

        index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
        index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];
        data_t a_value = matA_values[element_a];

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            index_t b_col = matB_col_ind[b_element] - start;
            data_t b_value = matB_values[b_element];
            
            // add to accumulator
            atomicAdd(&accumulator[b_col], a_value * b_value);

            // use atomicOr to set the bitmap
            index_t element_pos = (b_col) / 32;
            index_t bit_pos = (b_col) % 32;

            if (QUERY_BITMAP) {
                if ((bitmap[element_pos] & (1 << bit_pos)) == 0) {
                    atomicOr(&bitmap[element_pos], (1 << bit_pos));
                }
            } else {
                atomicOr(&bitmap[element_pos], (1 << bit_pos));
            }
        }
        //__syncwarp();
    }
    __syncthreads();

    auto ret = denseCompact<CHECK_OVERFLOW>(
        accumulator, accumulator_size, start,
        bitmap, prefix_sums,
        &out_cols[output_row_offsets[my_row]], &out_vals[output_row_offsets[my_row]],
        accumulator_size
    );

    if (CHECK_OVERFLOW) {
        if (ret == (index_t)(-1)) {
            if (tid == 0) {
                index_t pos = atomicAdd(out_num_overflow_rows, 1);
                out_overflow_row_ids[pos] = my_row;
                index_t add_num = num_products[my_row];
                if (add_num > num_cols_b) add_num = num_cols_b;
                index_t buffer_loc = atomicAdd(out_overflow_buffer_allocated, add_num);
                output_row_offsets[my_row] = buffer_loc;
                inout_my_row_ids[blockIdx.x] = (index_t)(-1);
            }
        }
    }

    if (out_nnz_per_row != nullptr) {
        if (tid == 0) {
            out_nnz_per_row[my_row] = ret;
        }
    }
}

// this kernel should work for a row with fewer than or equal to 32 threads
// however, it may not work for all cases
template<int LOG_NTHR, bool QUERY_BITMAP=true>
__device__ __forceinline__ void denseNumericSubWarpKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    data_t* accumulator,
    index_t* bitmap,
    index_t thread_id,
    index_t row_id,
    index_t start_col_id,
    index_t log_threads_per_b_row,
    index_t log_total_threads
) {
    const index_t my_group_id = thread_id >> log_threads_per_b_row;
    const index_t num_groups = 1 << (log_total_threads - log_threads_per_b_row);
    const index_t my_id_in_group = thread_id & ((1 << log_threads_per_b_row) - 1);
    const index_t group_size = 1 << log_threads_per_b_row;

    index_t start_element_a = matA_row_offsets[row_id];
    index_t end_element_a = (row_id+1 == num_rows_a) ? nnz_a: matA_row_offsets[row_id+1];

    for (index_t element_a = start_element_a + my_group_id;
         element_a < end_element_a;
         element_a += num_groups) {

        index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
        index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];
        data_t a_value = matA_values[element_a];

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            index_t b_col = matB_col_ind[b_element] - start_col_id;
            data_t b_value = matB_values[b_element];
            
            // add to accumulator
            atomicAdd(&accumulator[b_col], a_value * b_value);

            // use atomicOr to set the bitmap
            index_t element_pos = (b_col) / 32;
            index_t bit_pos = (b_col) % 32;

            if (QUERY_BITMAP) {
                if ((bitmap[element_pos] & (1 << bit_pos)) == 0) {
                    atomicOr(&bitmap[element_pos], (1 << bit_pos));
                }
            } else {
                atomicOr(&bitmap[element_pos], (1 << bit_pos));
            }
        }
    }
}


// This Kernel Should be called by exactly ONE warp
// Only the return value of the first element is guaranteed correct
__device__ __forceinline__ index_t denseCompactWarp(
    data_t* accumulator, index_t accumulator_size, index_t start_col_offset,
    uint32_t* bitmap, 
    index_t* idx_out, data_t* val_out
) {
    index_t previous_iters_sum = 0;
    const index_t my_tid = threadIdx.x % WARP_SIZE;
    const uint32_t mask = (1 << my_tid) - 1;

    int total_iters = (accumulator_size + WARP_SIZE - 1) / WARP_SIZE;
    for (int it = 0; it < total_iters; it ++ ) {

        if (it * WARP_SIZE + my_tid >= accumulator_size) {
            previous_iters_sum += __popc(bitmap[it]);
            break;
        }

        uint32_t masked_bitmap = bitmap[it] & mask;
        index_t my_loc_in_chunk = __popc(masked_bitmap);

        index_t my_output_loc = previous_iters_sum + my_loc_in_chunk;
        if ((bitmap[it] & (1 << my_tid)) != 0) {
            idx_out[my_output_loc] = start_col_offset + it * WARP_SIZE + my_tid;
            val_out[my_output_loc] = accumulator[it * WARP_SIZE + my_tid];
        }

        previous_iters_sum += __popc(bitmap[it]);
    }


    return previous_iters_sum;
}

// this kernel should guarantee no overflow.
// it should be called by exactly 64 threads
// Note that LOG_NTHR is the thread per subgroup. 
template<int LOG_NTHR, bool QUERY_BITMAP=true>
__global__ __launch_bounds__(WARP_SIZE*2, 2048>>6) void denseNumericSubWarpDispatcher(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *output_row_offsets, index_t *out_cols, data_t *out_vals,
    index_t *num_products, index_t *max_elements_b, index_t *start_element_b, index_t *end_element_b,
    index_t *my_row_ids,
    index_t num_rows,
    index_t accumulator_size,
    index_t *out_nnz_per_row=nullptr
) {

    // accumulator size is number of elements 
    extern __shared__ uint8_t dynamicShared[];

    const index_t group_size = 1 << LOG_NTHR;
    const index_t num_groups = WARP_SIZE * 2 / group_size;
    const index_t my_group_id = threadIdx.x / group_size;
    const index_t my_id_in_group = threadIdx.x % group_size;
    index_t shared_memory_per_group = accumulator_size * sizeof(data_t) + ((accumulator_size + 31) / 32) * sizeof(uint32_t); // needed shared memory size for each group, in bytes
    shared_memory_per_group = (shared_memory_per_group + 7) / 8 * 8;    // align to 8 bytes

    data_t* accumulator = (data_t*)&dynamicShared[my_group_id * shared_memory_per_group];
    uint32_t* bitmap = (uint32_t*)&accumulator[accumulator_size];

    index_t row_id = blockIdx.x * num_groups + (threadIdx.x / group_size);
    index_t my_row = (index_t)(-1);
    if (row_id < num_rows) {
        if (my_row_ids == nullptr) {
            my_row = row_id;
        } else {
            my_row = my_row_ids[row_id];
        }
    }

    for (index_t i = my_id_in_group; i < accumulator_size; i += group_size) {
        accumulator[i] = 0;
    }

    for (index_t i = my_id_in_group; i < (accumulator_size + 31) / 32; i += group_size) {
        bitmap[i] = 0;
    }

    __syncwarp();   // This should not be necessary *without* independent thread scheduling

    if (my_row != (index_t)(-1)) {
        index_t start = start_element_b[my_row];

        int log_nthr_per_a_elem = LOG_NTHR - 1;
        if (log_nthr_per_a_elem < 0) log_nthr_per_a_elem = 0;

        denseNumericSubWarpKernel<LOG_NTHR, QUERY_BITMAP>(
            matA_row_offsets, matA_col_ind, matA_values, num_rows_a, num_cols_a, nnz_a,
            matB_row_offsets, matB_col_ind, matB_values, num_rows_b, num_cols_b, nnz_b,
            accumulator,
            bitmap,
            my_id_in_group,
            my_row,
            start,
            log_nthr_per_a_elem,
            LOG_NTHR
        );

    }
    __syncthreads();

    const index_t warp_id = threadIdx.x / WARP_SIZE;
    index_t start_row = blockIdx.x * num_groups;
    index_t end_row = (blockIdx.x + 1) * num_groups;
    if (end_row > num_rows) end_row = num_rows;

    for (index_t row_in_block = warp_id; row_in_block < (end_row - start_row); row_in_block += blockDim.x / WARP_SIZE) {
        index_t actual_row;
        if (my_row_ids == nullptr) {
            actual_row = start_row + row_in_block;
        } else {
            actual_row = my_row_ids[start_row + row_in_block];
        }
        data_t* compact_accumulator = (data_t*)&dynamicShared[(row_in_block)*shared_memory_per_group];
        uint32_t* compact_bitmap = (uint32_t*)&compact_accumulator[accumulator_size];
        index_t start_col_id = start_element_b[actual_row];
        index_t end_col_id = end_element_b[actual_row];
        index_t valid_size = end_col_id - start_col_id + 1;

        index_t ret = denseCompactWarp(
            compact_accumulator,
            valid_size,
            start_col_id,
            compact_bitmap,
            &out_cols[output_row_offsets[actual_row]],
            &out_vals[output_row_offsets[actual_row]]
        );
        
        __syncwarp();

        if (out_nnz_per_row != nullptr) {
            if (threadIdx.x % WARP_SIZE == 0) {
                out_nnz_per_row[actual_row] = ret;
            }
        }
    }
}

// Used by the IterKernel, determine the starting point of next iteration
__device__ __forceinline__ index_t getNextStartingCol(index_t local_min, index_t *temp) {
    // first, do warp size min
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
    }
    if (threadIdx.x == 0) {
        *temp = (index_t)(-1);
    }
    __syncthreads();

    // use the first shared memory of temp as a temporary storage
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMin(temp, local_min);
    }
    __syncthreads();

    // return the result
    return *temp;
}

// Get one of the buffers allocated for this SM
// Could probably use the getGlobalBuffer instead
__device__ __forceinline__ index_t* getSMBufferLoc(
    index_t* buffer_row_locations, 
    uint32_t* buffer_bitmap,
    index_t num_rows_b,
    index_t** shared
) {
    if (threadIdx.x == 0) {
        auto my_sm_id = get_smid();
        int loc = -1;
        while(loc == -1) {
            int free_bit = __ffs(~buffer_bitmap[my_sm_id]) - 1;
            index_t old = atomicOr(&buffer_bitmap[my_sm_id], (1 << free_bit));
            if ((old & (1 << free_bit)) == 0) {
                loc = free_bit;
            }
        }
        if (loc >= LARGEST_KERNEL_CONCURRENT_BLOCKS) {
            printf("Error: Exceeded max concurrent rows per SM in dense numeric accumulator.\n");
        }
        // printf("BLOCK %d on SM %d allocated buffer location %d\n", blockIdx.x, my_sm_id, loc);
        (*shared) = &buffer_row_locations[(my_sm_id * LARGEST_KERNEL_CONCURRENT_BLOCKS + loc) * num_rows_b];
    }
    __syncthreads();

    return *shared;
}

__device__ __forceinline__ void releaseSMBufferLoc(
    index_t* buffer_row_locations, 
    index_t* my_buffer_loc,
    uint32_t* buffer_bitmap,
    index_t num_rows_b
) {
    if (threadIdx.x == 0) {
        auto my_sm_id = get_smid();
        index_t loc = (my_buffer_loc - &buffer_row_locations[my_sm_id * LARGEST_KERNEL_CONCURRENT_BLOCKS * num_rows_b]) / num_rows_b;
        // printf("BLOCK %d on SM %d released buffer location %d\n", blockIdx.x, my_sm_id, loc);
        atomicAnd(&buffer_bitmap[my_sm_id], ~(1 << loc));
    }
}

// This is the largest dense kernel. Would iterate over the entire row. May take multiple iterations.
template<int LOG_NTHR, bool CHECK_OVERFLOW=false, bool QUERY_BITMAP=true, class offset_t=index_t>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void denseNumericIterKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    offset_t *output_row_offsets, index_t *out_cols, data_t *out_vals,
    index_t *num_products, 
    index_t *max_elements_b, index_t *start_element_b, index_t *end_element_b,
    index_t *inout_my_row_ids,
    index_t accumulator_size,
    // index_t num_rows,
    index_t smem_row_offset_max,
    index_t *buffer_row_locations = nullptr,
    uint32_t *buffer_bitmap = nullptr,
    index_t *out_nnz_per_row=nullptr,
    index_t *output_row_max_size=nullptr,
    index_t *out_num_overflow_rows=nullptr,
    index_t *out_overflow_row_ids=nullptr,
    index_t *out_overflow_buffer_allocated=nullptr
) {
    // accumulator size is in number of elements, not bytes
    
    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    data_t* accumulator = (data_t*)&dynamicShared[0];
    uint32_t* bitmap = (uint32_t*)&dynamicShared[accumulator_size * sizeof(data_t)];
    uint32_t* prefix_sums = (uint32_t*)&bitmap[(accumulator_size + 31) / 32];

    index_t my_row;
    if (inout_my_row_ids == nullptr) {
        my_row = blockIdx.x;
    } else {
        my_row = inout_my_row_ids[blockIdx.x];
    }

    index_t start_element_a = matA_row_offsets[my_row];
    index_t end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];

    // load balance is still the same
    index_t log_nthr_per_a_elem = localLoadBalance(
        end_element_a - start_element_a,
        num_products[my_row],
        max_elements_b[my_row],
        2, LOG_NTHR 
    );

    index_t start_b_idx = start_element_b[my_row];
    index_t end_b_idx = end_element_b[my_row];
    
    // too many register use here; consider removing
    index_t my_group_id = threadIdx.x >> log_nthr_per_a_elem;
    index_t num_groups = blockDim.x >> log_nthr_per_a_elem;
    index_t my_id_in_group = threadIdx.x & ((1 << log_nthr_per_a_elem) - 1);
    index_t group_size = 1 << log_nthr_per_a_elem;


    // start map stores the starting location for each row of B
    index_t* start_map;

    if (end_element_a - start_element_a > smem_row_offset_max) {
        start_map = getSMBufferLoc(buffer_row_locations, buffer_bitmap, num_rows_b, (index_t**)accumulator);
    } else {
        start_map = (index_t*)&prefix_sums[(blockDim.x + 31) / 32 + 2];
    }

    for (index_t i = tid + start_element_a; i < end_element_a; i += blockDim.x) {
        auto b_row = matA_col_ind[i];
        start_map[i - start_element_a] = matB_row_offsets[b_row];
    }

    index_t output_element_offset = output_row_offsets[my_row];

    while (start_b_idx <= end_b_idx) {

        // init accumulator
        for (index_t i = tid; i < accumulator_size; i += blockDim.x) {
            accumulator[i] = 0;
        }
        for (index_t i = tid; i < (accumulator_size + 31) / 32; i += blockDim.x) {
            bitmap[i] = 0;
        }
        __syncthreads();
        index_t start_b_idx_next = (index_t)(-1);

        index_t total_iters = (end_element_a - start_element_a + num_groups -1) / num_groups;
        for (index_t i = 0; i < total_iters; i++) {

            index_t element_a = start_element_a + my_group_id + i * num_groups;
            index_t b_start;
            if (element_a < end_element_a) {
                b_start = start_map[element_a - start_element_a];
            }
            __syncthreads();    // we need to ensure start_map is not modified before use
            if (my_id_in_group == 0 && element_a < end_element_a) {
                start_map[element_a - start_element_a] = (index_t)(-1); //set to infinite
            }
            __syncthreads();

            if (element_a >= end_element_a) {
                break;  // all elements for the a row is processed
            }

            if (b_start == (index_t)(-1)) {
                continue;   // this a element is done
            }

            data_t a_value = matA_values[element_a];
            index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];    // this is in terms of loc in array; the other one is in term of idx

            for (index_t b_element = b_start + my_id_in_group;
                b_element < b_end;
                b_element += group_size) {

                index_t b_col = matB_col_ind[b_element] - start_b_idx;

                if (b_col >= accumulator_size) {
                    atomicMin(&start_map[element_a - start_element_a], b_element);
                    if (start_b_idx + b_col < start_b_idx_next) {
                        start_b_idx_next = start_b_idx + b_col;
                    }
                    break;
                }
                data_t b_value = matB_values[b_element];
                
                // add to accumulator
                atomicAdd(&accumulator[b_col], a_value * b_value);

                // use atomicOr to set the bitmap
                index_t element_pos = (b_col) / 32;
                index_t bit_pos = (b_col) % 32;

                if (QUERY_BITMAP) {
                    if ((bitmap[element_pos] & (1 << bit_pos)) == 0) {
                        atomicOr(&bitmap[element_pos], (1 << bit_pos));
                    }
                } else {
                    atomicOr(&bitmap[element_pos], (1 << bit_pos));
                }
            }
            //__syncwarp();
        }
        __syncthreads();

        index_t output_limit = index_t(-1);
        if (CHECK_OVERFLOW) {
            output_limit = output_row_max_size[my_row] + output_row_offsets[my_row] - output_element_offset;
        }

        index_t num_new_output = denseCompact<CHECK_OVERFLOW>(
            accumulator, accumulator_size, start_b_idx,
            bitmap, prefix_sums,
            &out_cols[output_element_offset], &out_vals[output_element_offset],
            output_limit
        );

        if (CHECK_OVERFLOW) {
            if (num_new_output == (index_t)(-1)) {
                if (tid == 0) {
                    // record the overflow row and change the buffer locs for next iter
                    index_t pos = atomicAdd(out_num_overflow_rows, 1);
                    out_overflow_row_ids[pos] = my_row;

                    index_t add_num = num_products[my_row];
                    if (add_num > num_cols_b) add_num = num_cols_b;
                    index_t buffer_loc = atomicAdd(out_overflow_buffer_allocated, add_num);
                    output_row_max_size[my_row] = add_num;
                    output_row_offsets[my_row] = buffer_loc;

                    if (inout_my_row_ids != nullptr) inout_my_row_ids[blockIdx.x] = (index_t)(-1);

                }
                break;
            }
        }

        output_element_offset += num_new_output;

        __syncthreads();
        start_b_idx = getNextStartingCol(start_b_idx_next, (index_t*)&dynamicShared[0]);
        __syncthreads();
            
    }

    if (out_nnz_per_row != nullptr) {
        if (tid == 0) {
            out_nnz_per_row[my_row] = output_element_offset - output_row_offsets[my_row];
        }
    }    

    start_element_a = matA_row_offsets[my_row];
    end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];

    if (end_element_a - start_element_a > smem_row_offset_max) {
        releaseSMBufferLoc(buffer_row_locations, start_map, buffer_bitmap, num_rows_b);
    }
    __syncthreads();
}

// This kernel uses fixed local load balance, and is used to handle some special matrices
template<int LOG_NTHR>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void denseNumericStaticKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t output_row_size, index_t *out_cols, data_t *out_vals,
    index_t accumulator_size,
    index_t log_nthr_per_a_elem,
    index_t *out_nnz_per_row=nullptr
    ) {
    
    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    data_t* accumulator = (data_t*)&dynamicShared[0];
    uint32_t* bitmap = (uint32_t*)&dynamicShared[accumulator_size * sizeof(data_t)];
    uint32_t* prefix_sums = (uint32_t*)&dynamicShared[accumulator_size * sizeof(data_t) + ((accumulator_size + 31) / 32) * sizeof(uint32_t)];

    index_t my_row = blockIdx.x;

    // init accumulator
    for (index_t i = tid; i < accumulator_size; i += blockDim.x) {
        accumulator[i] = 0;
    }

    for (index_t i = tid; i < (accumulator_size + 31) / 32; i += blockDim.x) {
        bitmap[i] = 0;
    }
    __syncthreads();
    
    index_t start_element_a = matA_row_offsets[my_row];
    index_t end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];

    // fixed load balance
    index_t my_group_id = threadIdx.x >> log_nthr_per_a_elem;
    index_t num_groups = blockDim.x >> log_nthr_per_a_elem;
    index_t my_id_in_group = threadIdx.x & ((1 << log_nthr_per_a_elem) - 1);
    index_t group_size = 1 << log_nthr_per_a_elem;

    for (index_t element_a = start_element_a + my_group_id;
         element_a < end_element_a;
         element_a += num_groups) {

        index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
        index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];
        data_t a_value = matA_values[element_a];

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            index_t b_col = matB_col_ind[b_element];
            data_t b_value = matB_values[b_element];
            
            // add to accumulator
            atomicAdd(&accumulator[b_col], a_value * b_value);

            // use atomicOr to set the bitmap
            index_t element_pos = (b_col) / 32;
            index_t bit_pos = (b_col) % 32;

            if (true) {
                if ((bitmap[element_pos] & (1 << bit_pos)) == 0) {
                    atomicOr(&bitmap[element_pos], (1 << bit_pos));
                }
            } else {
                atomicOr(&bitmap[element_pos], (1 << bit_pos));
            }
        }
        //__syncwarp();
    }
    __syncthreads();

    auto ret = denseCompact<false>(
        accumulator, accumulator_size, 0,
        bitmap, prefix_sums,
        &out_cols[output_row_size * my_row], &out_vals[output_row_size * my_row],
        accumulator_size
    );

    if (out_nnz_per_row != nullptr) {
        if (tid == 0) {
            out_nnz_per_row[my_row] = ret;
        }
    }
}

} // namespace somespgemm