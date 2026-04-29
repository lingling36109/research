#pragma once
#include "Common.h"

#include <limits>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

namespace somespgemm {


template<int NTHREADS>
__device__ __forceinline__ int countSortCompactFused(
    index_t key, data_t value,
    int subwarp_base, int subwarp_tidx,
    index_t* smem_key, data_t* smem_value,
    int product_num
) {
    // reuse smem_key and smem_value as scratch space
    index_t* scratch_index = smem_key;
    data_t* scratch_data = smem_value;
    bool leading = true;
    uint32_t bitmap = 0;

    if (subwarp_tidx < product_num) {
        scratch_index[subwarp_tidx] = key;
        scratch_data[subwarp_tidx] = value;
    } else {
        scratch_index[subwarp_tidx] = cuda::std::numeric_limits<index_t>::max();
        leading = false;
    }
    value = 0.0;

    __syncwarp();

    for (int k = 0; k < NTHREADS; k++ ) {
        if (scratch_index[k] == key) {
            value += scratch_data[k];
            if (k<subwarp_tidx) {
                leading = false;
            }
        }
        if (scratch_index[k] < key) {
            bitmap |= (1 << k);
        }
    }


    // use warp vote to get all leading elements
    // only for the subwarp
    auto leadings_bitmap = __ballot_sync(0xFFFFFFFF, leading);
    leadings_bitmap = (leadings_bitmap >> (subwarp_base % 32)) & ((1 << NTHREADS) - 1);
    auto n_elements = __popc(leadings_bitmap);
    bitmap &= leadings_bitmap;
    auto position = __popc(bitmap);

    if (leading) {
        smem_key[position] = key;
        smem_value[position] = value;
    }

    return n_elements;
}


// For ESC kernel, each thread would process only one element
template<int NTHREADS>
__device__ __forceinline__ void ESCKernel (
    index_t matA_st_idx, index_t matA_end_idx, index_t* matA_col_ind, data_t* matA_val,
    index_t* matB_row_offsets, index_t* matB_col_ind,  data_t* matB_val, 
    char* smem,
    volatile index_t* out_ind, volatile data_t* out_val, volatile index_t* out_num,
    index_t product_num, int subwarp_base, int subwarp_tidx
) {
    double val = 0.0;
    index_t c_col = MAX_INDEX;

    if (subwarp_tidx < product_num) {
        index_t current_a_idx = matA_st_idx;
        index_t current_b_row = matA_col_ind[current_a_idx];
        index_t current_c_remain = subwarp_tidx;

        index_t current_b_idx = 0;

        // find the element for this thread
        while( true ) {
            index_t current_row_start = matB_row_offsets[current_b_row];
            auto current_row_len = matB_row_offsets[current_b_row + 1] - current_row_start;
            if (current_row_len > current_c_remain) {
                current_b_idx = current_row_start + current_c_remain;
                break;
            }
            current_c_remain -= current_row_len;
            current_a_idx++;
            current_b_row = matA_col_ind[current_a_idx];
        }

        double a_val = matA_val[current_a_idx];
        double b_val = matB_val[current_b_idx];
        c_col = matB_col_ind[current_b_idx];
        val = a_val * b_val;
    }

    __syncwarp();

    auto smem_ind = (index_t*)smem;
    auto smem_val = (data_t*)(smem + sizeof(index_t) * NTHREADS);

    auto n_elements = countSortCompactFused<NTHREADS>(
        c_col, val, 
        subwarp_base, subwarp_tidx, 
        smem_ind, smem_val,
        product_num
    );

    __syncwarp();

    if (subwarp_tidx < n_elements) {
        out_ind[subwarp_tidx] = smem_ind[subwarp_tidx];
        out_val[subwarp_tidx] = smem_val[subwarp_tidx];
    }

    if (subwarp_tidx == 0) {
        *out_num = n_elements;
    }

}

template<int NTHREADS>
__global__ void ESCKernelDispatcher (
    index_t* matA_row_offsets, index_t* matA_col_ind, data_t* matA_val, size_t num_rows_A,
    index_t* matB_row_offsets, index_t* matB_col_ind, data_t* matB_val, size_t num_rows_B,
    index_t* temp_output_ind, data_t* temp_output_val, index_t* output_num,
    size_t* temp_output_offset, index_t* product_num, int rows_per_block
) {
    constexpr int num_subwarp = NUMERIC_USPARSE_K_BLOCK_SIZE / NTHREADS;

    int start_row = blockIdx.x * rows_per_block;
    int end_row = start_row + rows_per_block < num_rows_A ? start_row + rows_per_block : num_rows_A;

    int subwarp_id = threadIdx.x / NTHREADS;
    int subwarp_tidx = threadIdx.x % NTHREADS;
    int subwarp_base = subwarp_id * NTHREADS;

    __shared__ char smem[(NUMERIC_USPARSE_K_BLOCK_SIZE * (sizeof(data_t) + sizeof(index_t)) + sizeof(index_t) * num_subwarp)];
    auto my_smem = smem + (subwarp_base * (sizeof(data_t) + sizeof(index_t)));
    
    for (int i = start_row + subwarp_id; i < end_row; i += num_subwarp) {
        int matA_st = matA_row_offsets[i];
        int matA_ed = matA_row_offsets[i + 1];
        auto my_offset = temp_output_offset[i];
        if (product_num[i] > NTHREADS) {
            continue;
        }

        ESCKernel<NTHREADS>(
            matA_st, matA_ed, matA_col_ind, matA_val,
            matB_row_offsets, matB_col_ind, matB_val,
            my_smem,
            temp_output_ind + my_offset, temp_output_val + my_offset, output_num + i,
            product_num[i], subwarp_base, subwarp_tidx
        );  
        
    };
    
}

__global__ void classifyOutlierRows(
    index_t* product_num, index_t num_rows, int lower_threshold,
    index_t num_iters,
    index_t* out_outlier_rows, 
    index_t* out_outlier_count
) {
    int start_row = blockIdx.x * blockDim.x * num_iters + threadIdx.x;
    int end_row = start_row + blockDim.x * num_iters < num_rows ? start_row + blockDim.x * num_iters : num_rows;

    for (int i = start_row; i < end_row; i += blockDim.x) {
        if (product_num[i] > lower_threshold) {
            index_t my_pos = atomicAdd(out_outlier_count, 1);
            out_outlier_rows[my_pos] = i;
        }
    }
}

}
