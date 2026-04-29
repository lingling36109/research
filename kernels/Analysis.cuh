#pragma once
#include <cuda_runtime.h>
#include "Utils.h"
#include "Common.h"
#include "DeviceCommon.cuh"
#include "Hashmap.cuh"
namespace somespgemm {


int decideMatrixType (
    index_t A_nnz,
    index_t total_products,
    double avg_compaction,
    double avg_compaction2,
    Config& config
) {

    if (total_products / A_nnz < config.estimation_input_expansion_threshold) {
        return 0;
    }
    if (avg_compaction < config.estimation_output_compaction_threshold) {
        return 0;
    }
    if (avg_compaction2 < config.estimation_output_compaction_threshold) {
        return 0;
    }

    return 1;
}

template <int CONCURRENT_ROWS>
__global__ __launch_bounds__(PRODUCT_KERNEL_BLOCK_SIZE, MAX_THREADS_PER_SM/PRODUCT_KERNEL_BLOCK_SIZE) 
void analysisKernel (
    index_t *A_row_offsets, index_t *A_col_ids, size_t A_rows, size_t A_nnz,
    index_t *B_row_offsets, index_t *B_col_ids, size_t B_rows, size_t B_nnz,
    index_t *num_product,
    index_t *max_b_row_len,
    index_t *avg_b_row_len,
    index_t *b_col_idx_left,
    index_t *b_col_idx_right,
    index_t *max_a_row_nnz
) {
    constexpr int rows_per_block = PRODUCT_KERNEL_ITER_PER_BLOCK * CONCURRENT_ROWS;
    constexpr int threads_per_row = PRODUCT_KERNEL_BLOCK_SIZE / CONCURRENT_ROWS;
    int start_row = blockIdx.x * rows_per_block;
    int end_row = start_row + rows_per_block;
    if (end_row > A_rows) {
        end_row = A_rows;
    }

    size_t my_row_id = start_row + threadIdx.x / threads_per_row;
    int my_element_id = threadIdx.x % threads_per_row;

    index_t local_max_a_row_nnz = 0;
    
    for (int i = 0; i < PRODUCT_KERNEL_ITER_PER_BLOCK; i++) {
        if (my_row_id >= end_row) break;
        index_t row_start = A_row_offsets[my_row_id];
        index_t row_end = (my_row_id + 1 == A_rows) ? A_nnz : A_row_offsets[my_row_id + 1];
        index_t row_nnz = row_end - row_start;
        if (row_nnz > local_max_a_row_nnz) {
            local_max_a_row_nnz = row_nnz;
        }

        index_t product = 0;
        index_t max_b_len = 0;
        index_t b_left = (index_t)(-1);
        index_t b_right = 0;
        for (index_t j = row_start + my_element_id; j < row_end; j+= threads_per_row) {
            auto b_row = A_col_ids[j];
            auto b_row_st_element = B_row_offsets[b_row];
            auto b_row_ed_element = (b_row + 1 == B_rows) ? B_nnz : B_row_offsets[b_row + 1];
            if (b_row_ed_element <= b_row_st_element) continue;  // this will happen if the row is empty!
            auto b_element_count = b_row_ed_element - b_row_st_element;
            if (b_element_count > max_b_len) max_b_len = b_element_count;
            if (B_col_ids[b_row_st_element] < b_left) b_left = B_col_ids[b_row_st_element];
            if (B_col_ids[b_row_ed_element - 1] > b_right) b_right = B_col_ids[b_row_ed_element - 1];

            product += b_element_count;
        }

        // use cuda warp shuffle to reduce the result
        // first get active masks
        auto mask = __activemask();
        for (int offset = threads_per_row / 2; offset > 0; offset /= 2) {
            product += __shfl_down_sync(mask, product, offset, threads_per_row);
            max_b_len = max(max_b_len, __shfl_down_sync(mask, max_b_len, offset, threads_per_row));
            b_left = min(b_left, __shfl_down_sync(mask, b_left, offset, threads_per_row));
            b_right = max(b_right, __shfl_down_sync(mask, b_right, offset, threads_per_row));
        }
        if (my_element_id == 0) {
            num_product[my_row_id] = product;
            max_b_row_len[my_row_id] = max_b_len;
            avg_b_row_len[my_row_id] = (row_end - row_start) == 0 ? 0 : (product / (row_end - row_start));
            b_col_idx_left[my_row_id] = b_left;
            b_col_idx_right[my_row_id] = b_right;
        }

        my_row_id += CONCURRENT_ROWS;
    }

    if (local_max_a_row_nnz > *max_a_row_nnz) {
        atomicMax(max_a_row_nnz, local_max_a_row_nnz);
    }
}

// Currently One-pass binning is adopted
// So a full bin is needed for each item
// Could be optimized to a two-pass binning to reduce memory usage
template<bool EXPAND=false, bool COMPACT=false>
__global__ void binning (
    index_t *input,
    index_t *bin_count,
    index_t *bins,
    index_t *b_col_idx_left,
    index_t *b_col_idx_right,
    index_t num_items,
    index_t items_per_block,
    double expand_factor=0.0,
    double avg_compaction=0.0
) {
    __shared__ index_t local_bin_count[BINNING_K_WARP_PER_BLOCK * BIN_NUM * 2];
    __shared__ index_t local_input_ids[BINNING_K_BLOCK_SIZE * BIN_NUM * 2];

    auto warp_id = threadIdx.x / WARP_SIZE;
    auto lane_id = threadIdx.x % WARP_SIZE;
    auto my_bin_count = &local_bin_count[warp_id * BIN_NUM * 2];

    index_t current_item = blockIdx.x * items_per_block + threadIdx.x;
    index_t end_item = (blockIdx.x + 1) * items_per_block;
    index_t num_iters = (items_per_block + BINNING_K_BLOCK_SIZE - 1) / BINNING_K_BLOCK_SIZE;

    for (auto iter = 0; iter < num_iters; iter++) {
        // clear bin count
        if (lane_id < BIN_NUM * 2) {
            my_bin_count[lane_id] = 0;
        }
        __syncwarp();

        auto range = b_col_idx_right[current_item] - b_col_idx_left[current_item] + 1;
        auto dense_bin_id = BIN_NUM;
        for (int b = 0; b < BIN_NUM - 1; b++) {
            if (range <= DENSE_BIN_SIZES_GPU[b]) {
                dense_bin_id = b;
                break;
            }
        }

        auto val = current_item < num_items ? input[current_item] : 0;
        if (COMPACT) {
            val = static_cast<index_t>(val / avg_compaction);
        }
        if (EXPAND) {
            val = static_cast<index_t>(val * expand_factor);
        }

        auto bin_id = BIN_NUM - 1;
        for (int b = 0; b < BIN_NUM - 1; b++) {
            if (val <= HASH_BIN_SIZES_GPU[b]) {
                bin_id = b;
                break;
            }
        }
        
        // speck style design
        if (dense_bin_id <= bin_id) {
            // we are using bin_id here, even for dense kernel, because bin_id signifies the workload
            if (bin_id != BIN_NUM - 1) {
                bin_id = bin_id + BIN_NUM;
            } else {
                bin_id = bin_id + BIN_NUM - 1;
            }
        }

        // plain design
        /*
        if (dense_bin_id <= bin_id) {
            bin_id = dense_bin_id + BIN_NUM;
        }
        */

        // new design A
        /*
        if (dense_bin_id <= bin_id) {
            bin_id = (bin_id + dense_bin_id) / 2 + BIN_NUM;
        }
        */


        index_t local_idx = 0;
        if (current_item < num_items) {
            local_idx = atomicAdd(&my_bin_count[bin_id], 1);
        }
    
        __syncwarp();


        if (lane_id < BIN_NUM * 2) {
            auto start = atomicAdd(&bin_count[lane_id], my_bin_count[lane_id]);
            my_bin_count[lane_id] = start;
        }
        __syncwarp();
        auto global_idx = my_bin_count[bin_id] + local_idx;

        if (current_item < num_items) {
            bins[bin_id * num_items + global_idx] = current_item;
        }

        current_item += BINNING_K_BLOCK_SIZE;
    }
}

// this is similar to the above one,
template<bool EXPAND=false, bool MODIFY_DENSE_KERNEL_EST_VALUE=false>
__global__ void numericBinning (
    index_t *input,
    index_t *bin_count,
    index_t *bins,
    index_t *b_col_idx_left,
    index_t *b_col_idx_right,
    index_t num_items,
    index_t items_per_block,
    double expand_factor=0.0,
    bool use_numeric_largest=false,
    index_t* hybrid_hashmap_cnt=nullptr,
    index_t hybrid_hashmap_max=0
) {
    __shared__ index_t local_bin_count[BINNING_K_WARP_PER_BLOCK * BIN_NUM * 2];

    auto warp_id = threadIdx.x / WARP_SIZE;
    auto lane_id = threadIdx.x % WARP_SIZE;
    auto my_bin_count = &local_bin_count[warp_id * BIN_NUM * 2];

    index_t current_item = blockIdx.x * items_per_block + threadIdx.x;
    index_t end_item = (blockIdx.x + 1) * items_per_block;
    index_t num_iters = (items_per_block + BINNING_K_BLOCK_SIZE - 1) / BINNING_K_BLOCK_SIZE;

    for (auto iter = 0; iter < num_iters; iter++) {
        // clear bin count
        if (lane_id < BIN_NUM * 2) {
            my_bin_count[lane_id] = 0;
        }
        __syncwarp();

        auto range = b_col_idx_right[current_item] - b_col_idx_left[current_item] + 1;
        auto dense_bin_id = BIN_NUM - 1;
        for (int b = 0; b < BIN_NUM - 1; b++) {
            if (range <= DENSE_NUMERIC_BIN_SIZES_GPU[b]) {
                dense_bin_id = b;
                if (b == 0 && MODIFY_DENSE_KERNEL_EST_VALUE) {
                    input[current_item] = DENSE_NUMERIC_BIN_SIZES_GPU[0];
                }
                break;
            }
        }

        auto val = current_item < num_items ? input[current_item] : 0;
        if (EXPAND) {
            val = static_cast<index_t>(val * expand_factor);
        }

        int numeric_max_bin = use_numeric_largest && (*hybrid_hashmap_cnt < hybrid_hashmap_max) ? BIN_NUM : BIN_NUM - 1;
        auto bin_id = BIN_NUM;
        // Smallest bin is not supported for numerical now
        for (int b = 1; b < numeric_max_bin; b++) {
            if (val <= HASH_NUMERIC_BIN_SIZES_GPU[b]) {
                bin_id = b;
                break;
            }
        }

        // prioritize dense; except for the hybrid hash-based kernel (largest)
        if (dense_bin_id <= bin_id && !(bin_id == BIN_NUM - 1 && dense_bin_id == BIN_NUM - 1)) {
            bin_id = dense_bin_id + BIN_NUM;
        }

        // limit the number of hybrid hash-based kernel
        if (bin_id == BIN_NUM-1) {
            atomicAdd(hybrid_hashmap_cnt, 1);
        }


        index_t local_idx = 0;
        if (current_item < num_items) {
            local_idx = atomicAdd(&my_bin_count[bin_id], 1);
        }
    
        __syncwarp();

        if (lane_id < BIN_NUM * 2) {
            auto start = atomicAdd(&bin_count[lane_id], my_bin_count[lane_id]);
            my_bin_count[lane_id] = start;
        }
        __syncwarp();
        auto global_idx = my_bin_count[bin_id] + local_idx;

        if (current_item < num_items) {
            bins[bin_id * num_items + global_idx] = current_item;
        }

        current_item += BINNING_K_BLOCK_SIZE;
    }
}

}