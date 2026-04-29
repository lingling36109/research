#pragma once
#include "MurmurHash.cuh"
#include "DeviceCommon.cuh"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace somespgemm {

// uint32_t is used for scratch because CUDA does not support atomic operations on uint8_t.
template<int logPrecision>
__device__ __forceinline__ void hllAdd(
    uint32_t hashed_value,
    uint32_t* scratch
    ) {
    constexpr int precision = 1 << logPrecision;
    constexpr uint32_t mask = (1 << logPrecision) - 1;
    int idx = int(hashed_value&mask);
    uint32_t item = __clz(hashed_value) + 1;
    if (item > ((32 - logPrecision) + 1)) {
        item = (32 - logPrecision) + 1;
    }
    atomicMax(&scratch[idx], item);
}

// This kernel is limited to 32 threads (1 warp) per block. 
// May not be the best efficiency
template<int logPrecision>
__global__ void hllConstruct(
    index_t *matB_row_offsets, 
    index_t *matB_col_ind, 
    index_t num_rows_b, 
    index_t num_cols_b, 
    index_t nnz_b,
    uint8_t* global_hll,
    int num_sections,
    int rows_per_block
) {
    constexpr int precision = 1 << logPrecision;
    index_t row = blockIdx.x * rows_per_block;
    index_t end_row = (row + rows_per_block) < num_rows_b ? (row + rows_per_block) : num_rows_b;
    index_t start_element = matB_row_offsets[row];
    index_t end_element = (end_row < num_rows_b) ? matB_row_offsets[end_row] : nnz_b;
    extern __shared__ uint32_t smem[];

    int items_per_row = precision * num_sections;
    int items_total = items_per_row * rows_per_block;
    index_t col_per_section = (num_cols_b + num_sections - 1) / num_sections;

    uint32_t* current_scratch = smem;
    index_t* selected_row_offsets = (index_t*)(smem + items_total);

    index_t current_row = 0;
    index_t current_max_col = col_per_section;

    // init scratch
    for (int i = threadIdx.x; i < items_total; i += blockDim.x) {
        smem[i] = 0;
    }

    // init selected_row_offsets
    int num_rows_in_block = end_row - row;
    for (int i = threadIdx.x; i < num_rows_in_block; i += blockDim.x) {
        selected_row_offsets[i] = matB_row_offsets[row + i];
    }
    // Add sentinel value for bounds checking
    if (threadIdx.x == 0 && num_rows_in_block > 0) {
        selected_row_offsets[num_rows_in_block] = end_element;
    }

    __syncthreads();

    // Process each element in the row
    for (index_t i = start_element + threadIdx.x; i < end_element; i += blockDim.x) {
        index_t col = matB_col_ind[i];

        while (i >= selected_row_offsets[current_row+1]) {
            current_row++;
            current_scratch = smem + current_row * items_per_row;
            current_max_col = col_per_section;
        }

        // find the row that contains the column
        while (col >= current_max_col) {
            current_max_col += col_per_section;
            current_scratch += precision;
        }

        uint32_t hashed_value = murmurHash3_32(col);
        hllAdd<logPrecision>(hashed_value, current_scratch);
    }

    __syncthreads();

    // shrink from shared memory to global memory
    uint8_t* global_ptr = global_hll + row * items_per_row;
    int total_items_to_copy = items_per_row * num_rows_in_block;
    for (int i = threadIdx.x; i < total_items_to_copy; i += blockDim.x) {
        global_ptr[i] = static_cast<uint8_t>(smem[i]);
    }
    __syncthreads();    

}


// This function requires items_per_row >= 4 * blockdim.x
template<int logPrecision, bool UP_THRESHOLD=false, bool DOWN_THRESHOLD=false>
__global__ void hllMerge(
    index_t *matA_row_offsets, 
    index_t *matA_col_ind, 
    index_t num_rows_a, 
    index_t num_cols_a, 
    index_t nnz_a,
    uint8_t* b_hll,
    uint8_t* c_hll,
    index_t* c_estimate,
    double hll_expansion_factor=1.0,
    index_t threshold = 0
) {
    constexpr int precision = 1 << logPrecision;
    constexpr int bytes_per_thread = HLL_MERGE_K_BYTES_PER_THREAD;
    constexpr double a = HLL_CONSTANT[logPrecision];
    using read_t = uint32_t;
    const int items_per_row = precision;

    index_t row = blockIdx.x;
    index_t start_element = matA_row_offsets[row];
    index_t end_element = row == num_rows_a ? nnz_a : matA_row_offsets[row+1];

    if (UP_THRESHOLD) {
        if (end_element - start_element >= threshold) {
            return;
        }
    }
    if (DOWN_THRESHOLD) {
        if (end_element - start_element < threshold) {
            return;
        }
    }

    int elements_per_iter = blockDim.x * bytes_per_thread;
    int b_rows_per_iter = elements_per_iter / items_per_row;
    int my_offset_row_b = threadIdx.x / (blockDim.x / b_rows_per_iter);
    int my_offset_col_b = bytes_per_thread * (threadIdx.x % (blockDim.x / b_rows_per_iter));

    int reduction_lane_id = threadIdx.x % precision;

    extern __shared__ uint32_t scratch[];
    uint8_t* smem = (uint8_t*)scratch;
    
    if (elements_per_iter % items_per_row != 0) {
        // This should not happen
        printf("Error: elements_per_iter %% items_per_row != 0\n");
        return;
    }

    uint32_t max_value_packed = 0;

    // The prefetch version and it's not working well
    // core merge
    for (index_t i = start_element + my_offset_row_b; i < end_element; i += b_rows_per_iter) {
        auto row_b = matA_col_ind[i];
        auto element_b_idx = row_b * items_per_row + my_offset_col_b;
        
        read_t buf = *(read_t*)(b_hll + element_b_idx);
        max_value_packed = __vmaxu4(max_value_packed, buf);
    }

    // result write back. sync through shared mem
    #pragma unroll
    for (int i = 0; i < bytes_per_thread; i++) {
        // smem[threadIdx.x * bytes_per_thread + i] = max_value[i];
        smem[threadIdx.x * bytes_per_thread + i] = static_cast<uint8_t>(max_value_packed & 0xFF);
        max_value_packed = max_value_packed >> 8;
    }
    __syncthreads();

    // estimation calculation
    int items_padded = items_per_row < 32 ? 32 : items_per_row;
    double z = 0.0;
    int empty_reg = 0;
    for (int i = threadIdx.x; i < items_padded; i += blockDim.x) {

        uint8_t max_value = 0;

        if (i < items_per_row) {
            for (int j = 0; j < b_rows_per_iter; j++) {
                if (max_value < smem[j*items_per_row+i]) {
                    max_value = smem[j*items_per_row+i];
                }
            }
        }
        z += (double)1.0 / (1ULL << max_value);
        empty_reg += (max_value == 0 ? 1 : 0);
    }

    // precision = 32 then start at 16
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        z += __shfl_down_sync(0xFFFFFFFF, z, offset);
        empty_reg += __shfl_down_sync(0xFFFFFFFF, empty_reg, offset);
    }
        
    __shared__ double total_z;
    __shared__ int total_empty_reg;
    if (threadIdx.x == 0) {
        total_z = 0.0;
        total_empty_reg = 0;
    }
    __syncthreads();

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&total_z, z);
        atomicAdd(&total_empty_reg, empty_reg);
    }
    __syncthreads();
        
    if (threadIdx.x == 0) {
        z = total_z;
        empty_reg = total_empty_reg;
        
        int result = static_cast<int>(a * precision * precision / z);
        if (empty_reg != 0 && result < 2.5 * precision) {
            result = static_cast<int>(precision*(log(precision/(float)empty_reg)));
        }

        index_t temp = (result * hll_expansion_factor);
        for (int b = 0; b < BIN_NUM-1; b++) {
            if (temp <= HASH_NUMERIC_BIN_SIZES_GPU[b]) {
                temp = HASH_NUMERIC_BIN_SIZES_GPU[b];
                break;
            }
        }
        c_estimate[row] = temp;

    }
}


template<int logPrecision>
__global__ __launch_bounds__(HLL_SAMPLE_K_THREADS_PER_BLOCK, 2048/HLL_SAMPLE_K_THREADS_PER_BLOCK) void hllSamplingMerge(
    index_t* matA_row_offsets, index_t* matA_col_ind, index_t num_rows_a, index_t nnz_a,
    uint8_t* b_hll,
    index_t* num_products,
    index_t* out_sample_total,
    index_t* out_product_total,
    double* out_compaction_total,
    double* out_sampled_compaction
) {

    constexpr int precision = 1 << logPrecision;
    constexpr int bytes_per_thread = 4;
    constexpr double a = HLL_CONSTANT[logPrecision];
    using read_t = uint32_t;
    const int items_per_row = precision;

    int sample_idx = blockIdx.x;
    const uint64_t lcg_a = 1664525;
    const uint64_t lcg_c = 1013904223;
    const uint64_t lcg_m = (1ULL << 32);
    uint64_t x = 112233 + sample_idx * 2654435761ULL;
    x = (lcg_a * x + lcg_c) % lcg_m;
    index_t row = x % num_rows_a;

    index_t start_element = matA_row_offsets[row];
    index_t end_element = row == num_rows_a ? nnz_a : matA_row_offsets[row+1];

    int elements_per_iter = blockDim.x * bytes_per_thread;
    int b_rows_per_iter = elements_per_iter / items_per_row;
    int my_offset_row_b = threadIdx.x / (blockDim.x / b_rows_per_iter);
    int my_offset_col_b = bytes_per_thread * (threadIdx.x % (blockDim.x / b_rows_per_iter));

    int reduction_lane_id = threadIdx.x % precision;

    extern __shared__ uint32_t scratch[];
    uint8_t* smem = (uint8_t*)scratch;
    
    if (elements_per_iter % items_per_row != 0) {
        return;
    }

    // this should be kept in reg.
    uint32_t max_value_packed = 0;


    // core merge
    for (index_t i = start_element + my_offset_row_b; i < end_element; i += b_rows_per_iter) {
        auto row_b = matA_col_ind[i];
        auto element_b_idx = row_b * items_per_row + my_offset_col_b;
        
        read_t buf = *(read_t*)(b_hll + element_b_idx);
        max_value_packed = __vmaxu4(max_value_packed, buf);
    }


    #pragma unroll
    for (int i = 0; i < bytes_per_thread; i++) {
        smem[threadIdx.x * bytes_per_thread + i] = static_cast<uint8_t>(max_value_packed & 0xFF);
        max_value_packed = max_value_packed >> 8;
    }
    __syncthreads();

    // estimation calculation
    int items_padded = items_per_row < 32 ? 32 : items_per_row;
    double z = 0.0;
    int empty_reg = 0;
    for (int i = threadIdx.x; i < items_padded; i += blockDim.x) {

        uint8_t max_value = 0;

        if (i < items_per_row) {
            for (int j = 0; j < b_rows_per_iter; j++) {
                if (max_value < smem[j*items_per_row+i]) {
                    max_value = smem[j*items_per_row+i];
                }
            }
        }
        z += (double)1.0 / (1ULL << max_value);
        empty_reg += (max_value == 0 ? 1 : 0);
    }


    // precision = 32 then start at 16
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        z += __shfl_down_sync(0xFFFFFFFF, z, offset);
        empty_reg += __shfl_down_sync(0xFFFFFFFF, empty_reg, offset);
    }
        
    __shared__ double total_z;
    __shared__ int total_empty_reg;
    if (threadIdx.x == 0) {
        total_z = 0.0;
        total_empty_reg = 0;
    }
    __syncthreads();

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&total_z, z);
        atomicAdd(&total_empty_reg, empty_reg);
    }
    __syncthreads();
    // this is not necessary for now. we only call with one warp
        
    if (threadIdx.x == 0) {
        z = total_z;
        empty_reg = total_empty_reg;
        int result = static_cast<int>(a * precision * precision / z);
        if (empty_reg != 0 && result < 2.5 * precision) {
            result = static_cast<int>(precision*(log(precision/(float)empty_reg)));
        }

        atomicAdd(out_sample_total, result);
        atomicAdd(out_product_total, (num_products[row]));
        atomicAdd(out_compaction_total, (double)(num_products[row]) / (double)result);
        out_sampled_compaction[blockIdx.x] = (double)(num_products[row]) / (double)result;
    }
    
}

}
