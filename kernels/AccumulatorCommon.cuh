#pragma once
#include "Common.h"
#include <cuda_runtime.h>

namespace somespgemm {
    

__device__ __forceinline__ int getGlobalBuffer(uint32_t* global_buffer_bitmap, int num_buffers) {
    auto current_chunk = blockIdx.x % (num_buffers / 32);
    auto num_chunks = num_buffers / 32;
    while (true) {
        uint32_t chunk = global_buffer_bitmap[current_chunk];
        if ((chunk != 0xFFFFFFFF)) {
            // There is at least one free buffer in this chunk
            int bit_pos = __ffs(~chunk) - 1; // Find first zero bit
            if (bit_pos >= 0) {
                auto old = atomicOr(&global_buffer_bitmap[current_chunk], (1U << bit_pos));
                // Check if we successfully claimed the bit
                if ((old & (1U << bit_pos)) == 0) {
                    index_t buffer_idx = current_chunk * 32 + bit_pos;
                    // printf("Block %d Acquired global buffer %d from chunk %d\n", blockIdx.x, buffer_idx, current_chunk);
                    return buffer_idx;  
                }
            }
        }

        current_chunk++;
        if (current_chunk >= num_chunks) {
            current_chunk = 0;
        }
    }
}

__device__ __forceinline__ void releaseGlobalBuffer(uint32_t* global_buffer_bitmap, int my_buffer_idx) {
    int chunk_idx = my_buffer_idx / 32;
    int bit_pos = my_buffer_idx % 32;
    auto old = atomicAnd(&global_buffer_bitmap[chunk_idx], ~(1U << bit_pos));
}

// This is to ensure that, threads in a block get the same buffer
// We do not want to sync the block
__device__ __forceinline__ index_t getUniqueGlobalBuffer(
    index_t *block_global_map_idx,
    uint32_t *global_buffer_bitmap,
    int num_buffers
) {

    // __nanosleep(1); not supported on some hardware

    if (block_global_map_idx[0] != (index_t)(-1)) {
        return block_global_map_idx[0];
    }

    index_t buf_idx = getGlobalBuffer(global_buffer_bitmap, num_buffers);
    auto old = atomicCAS(block_global_map_idx, (index_t)(-1), buf_idx);
    if (old == (index_t)(-1)) {
        return buf_idx;
    } else {
        releaseGlobalBuffer(global_buffer_bitmap, buf_idx);
        return old;
    }
    
}


// Basically following speck
__device__ __forceinline__ uint32_t localLoadBalance(index_t n_element_a, index_t num_products, index_t max_elements_b, uint32_t lbound, uint32_t ubound) {
    // simple linear mapping for now
    if (num_products == 0) return 0; // should not happen
    // const index_t total_threads = 1 << ubound; 
    // It should be satisfied that total_threads = 1 << ubound

    // using the avg number of products as a starting point. excl. the largest b element
    index_t avg_ops = max(1U, (num_products - max_elements_b) / max(1U, n_element_a - 1));
    index_t log_nthr = max(1U, (31 - __clz(avg_ops)));
    
    // we want to find the closet elem, not just cutting off, so here's an adjustment
    if ((1U << log_nthr) * 3 < avg_ops * 2) { log_nthr += 1; }

    index_t a_elem_per_iter = 1 << (ubound - log_nthr);
    index_t num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    index_t max_sub_iter = (max_elements_b + (1<<log_nthr) - 1) >> log_nthr;

    while (max_sub_iter > num_iters * 2) {
        if (log_nthr >= ubound) break;
        log_nthr += 1;
        max_sub_iter = (max_elements_b + (1<<log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }

    while (num_iters > max_sub_iter * 2) {
        if (log_nthr <= lbound) break;
        log_nthr -= 1;
        max_sub_iter = (max_elements_b + (1<<log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }

    while (a_elem_per_iter > n_element_a && (1 << log_nthr) < max_elements_b) {
        log_nthr += 1;
        a_elem_per_iter = 1 << (ubound - log_nthr);
    }


    if (log_nthr < lbound) log_nthr = lbound;
    if (log_nthr > ubound) log_nthr = ubound;

    return log_nthr;
}

} // namespace somespgemm