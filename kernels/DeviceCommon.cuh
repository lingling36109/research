#pragma once
#include "Common.h"
#include <cuda_runtime.h>

namespace somespgemm{

__device__ __constant__ uint32_t HASH_BIN_SIZES_GPU[BIN_NUM] = {
    HASH_BIN_SIZES[0],
    HASH_BIN_SIZES[1],
    HASH_BIN_SIZES[2],
    HASH_BIN_SIZES[3],
    HASH_BIN_SIZES[4],
    HASH_BIN_SIZES[5],
    HASH_BIN_SIZES[6]
};

__device__ __constant__ uint32_t DENSE_BIN_SIZES_GPU[BIN_NUM] = {
    DENSE_BIN_SIZES[0],
    DENSE_BIN_SIZES[1],
    DENSE_BIN_SIZES[2],
    DENSE_BIN_SIZES[3],
    DENSE_BIN_SIZES[4],
    DENSE_BIN_SIZES[5],
    DENSE_BIN_SIZES[6]
};

__device__ __constant__ uint32_t HASH_NUMERIC_BIN_SIZES_GPU[BIN_NUM] = {
    HASH_NUMERIC_BIN_SIZES[0],
    HASH_NUMERIC_BIN_SIZES[1],
    HASH_NUMERIC_BIN_SIZES[2],
    HASH_NUMERIC_BIN_SIZES[3],
    HASH_NUMERIC_BIN_SIZES[4],
    HASH_NUMERIC_BIN_SIZES[5],
    HASH_NUMERIC_BIN_SIZES[6]
};

#define NUM_HASH_INPLACE_SORT_THRESHOLD HASH_NUMERIC_BIN_SIZES_GPU[2]

__device__ __constant__ uint32_t DENSE_NUMERIC_BIN_SIZES_GPU[BIN_NUM] = {
    DENSE_NUMERIC_BIN_SIZES[0],
    DENSE_NUMERIC_BIN_SIZES[1],
    DENSE_NUMERIC_BIN_SIZES[2],
    DENSE_NUMERIC_BIN_SIZES[3],
    DENSE_NUMERIC_BIN_SIZES[4],
    DENSE_NUMERIC_BIN_SIZES[5],
    DENSE_NUMERIC_BIN_SIZES[6]
};

#define CHECK_AND_FREE_DEVICE_MEM(ptr) \
    if (ptr != nullptr) { \
        CHECK_CUDA(cudaFree(ptr)); \
        ptr = nullptr; \
    }

#define CHECK_AND_FREE_DEVICE_MEM_ASYNC(ptr, stream) \
    if (ptr != nullptr) { \
        CHECK_CUDA(cudaFreeAsync(ptr, stream)); \
        ptr = nullptr; \
    }

static __device__ __inline__ unsigned int get_smid(void) {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}


} // namespace somespgemm