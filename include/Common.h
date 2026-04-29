#pragma once

#include <cstdint>
#include <climits>

// index_t should be either uint32_t or uint64_t. can not be signed type
typedef uint32_t index_t;
typedef double data_t;

//
namespace somespgemm {

/* Hardware related params */
#define NUM_SM 108
#define SHARED_MEMORY_KB 128  /* Choices: 128, 96, 80 . HAS TO BE ONE OF THEM*/

#define WARP_SIZE 32
#define MAX_THREADS_PER_SM 2048    // MAX Threads per SM

/* Output params */
#define BANNER "================================================================"

/* Output checker */
#define COMPARE_RELATIVE_ERROR 1e-6
#define COMPARE_RELATIVE_ERROR_NO_TOLERATE 1e-4

/* CUDA kernel params*/
#define HLL_K_CONSTRUCT_NBLOCKS 16384
#define HLL_K_CONSTRUCT_MIN_ROWS_PER_BLOCK 1
#define HLL_K_CONSTRUCT_MAX_ROWS_PER_BLOCK 32

#define HLL_MERGE_MULTI_KERNEL_THRESHOLD 8 // if max / avg > this, use a heavy kernel
#define HLL_MERGE_K_BYTES_PER_THREAD 4

#define HLL_SAMPLE_K_THREADS_PER_BLOCK 128
#define HLL_SAMPLE_MAX_OPS_TIME 5

#define BINNING_K_WARP_PER_BLOCK 2
#define BINNING_K_BLOCK_SIZE (WARP_SIZE * BINNING_K_WARP_PER_BLOCK)
#define BINNING_K_ITER_PER_BLOCK 8

#define PRODUCT_KERNEL_BLOCK_SIZE 64
#define PRODUCT_KERNEL_ITER_PER_BLOCK 8

#define NUMERIC_USPARSE_K_BLOCK_SIZE 64
#define NUMERIC_USPARSE_K_ROWS_PER_BLOCK 8  // may change that to be consistent

#define NUMERIC_HASH_K_BLOCK_SIZE 64

/* Scientific and decision  params */
constexpr double HLL_CONSTANT[8] = {0.0, 0.0, 0.0, 0.0, 0.673, 0.697, 0.709, 0.715};

#define BIN_NUM 7

#define HASH_SYMBOLIC_GMEM_INIT_SWITCH_COE 1.5
#define HASH_SYMBOLIC_GMEM_SIZE_EXPAND_COE 1.25

// These two are hashmap full threshold
#define HLL_HASHMAP_MAX_LOAD_FACTOR 0.8
#define HASH_SYMBOLIC_GMEM_FULL_COE 2

#define LARGEST_KERNEL_NUM_BLOCKS NUM_SM * 2
#define LARGEST_KERNEL_NUM_BUFFERS NUM_SM * 3
#define LARGEST_KERNEL_CONCURRENT_BLOCKS 2

#define DENSE_NUMERIC_SMALLEST_LOG_NTHR 5

constexpr int NUMERIC_USPARSE_KERNEL_SIZES[] = {16, 32, 64, 128}; // do we want 2, 4, 8 kernel sizes? 

/* Constants */
#define HASH_UNUSED 0xFFFFFFFF
#define MAX_INDEX 0xFFFFFFFF
#define MURMUR3_SEED 1234


// The first two kernels have the same number of concurrent blocks; however the first have two independent warps
// The last two kernels have the same number of concurrent blocks; however the last may use global memory
constexpr int CONCURRENT_BLOCKS_PER_SM[] = {32, 32, 16, 8, 4, 2, 2};   
constexpr int BLOCK_SIZES[] = {
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[0],
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[1],
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[2],
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[3],
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[4],
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[5],
    MAX_THREADS_PER_SM / CONCURRENT_BLOCKS_PER_SM[6]
};

constexpr uint32_t DENSE_NUMERIC_LARGEST_BLOCK_SIZE = BLOCK_SIZES[6];

constexpr int bits_needed(unsigned int x) {
    int bits = 0;
    while (x) {
        x >>= 1;
        ++bits;
    }
    return bits;
}

#define DYN_SHARED_MEM_PER_SM SHARED_MEMORY_KB * 1024  // Total Usable Shared Memory per SM in bytes. Too high value may reduce L1 cache to too small.

constexpr int SHARED_MEM_PER_BLOCK[] = {
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[0],
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[1],
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[2],
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[3],
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[4],
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[5],
    DYN_SHARED_MEM_PER_SM / CONCURRENT_BLOCKS_PER_SM[6]
};


#if SHARED_MEMORY_KB == 128
// Hard-coded prime bin sizes for 128KB Shared Memory
constexpr uint32_t HASH_BIN_SIZES[] = {
    503,
    1013,
    2029,
    4079,
    8179,
    16369,
    16369
};

constexpr uint32_t HASH_NUMERIC_BIN_SIZES[] = {
    0,
    331,
    673,
    1327,
    2719,
    5449,
    16253
};

// setup 1 for 128KB shared memory per block
constexpr uint32_t SORT_ITEMS_PER_THREAD[] = {
    0,
    4,
    8,
    16,
    16,
    16,
    16
};

constexpr uint32_t SORT_BLOCK_SIZES[] = {
    0,
    32 * 3,
    32 * 3,
    32 * 3,
    32 * 6,
    32 * 11,
    32 * 32
};

#elif SHARED_MEMORY_KB == 96

constexpr uint32_t HASH_BIN_SIZES[] = {
    373,
    757,
    1523,
    3061,
    6133,
    12277,
    12277
};

constexpr uint32_t HASH_NUMERIC_BIN_SIZES[] = {
    0,
    241,
    499,
    1013,
    2029,
    4079,
    12277
};

// setup 1 for 128KB shared memory per block
constexpr uint32_t SORT_ITEMS_PER_THREAD[] = {
    0,
    4,
    8,
    16,
    16,
    16,
    16
};

constexpr uint32_t SORT_BLOCK_SIZES[] = {
    0,
    32 * 2,
    32 * 2,
    32 * 2,
    32 * 4,
    32 * 8,
    32 * 24
};

#elif SHARED_MEMORY_KB == 80

constexpr uint32_t HASH_BIN_SIZES[] = {
    307,
    619,
    1259,
    2549,
    5107,
    10223,
    10223
};

constexpr uint32_t HASH_NUMERIC_BIN_SIZES[] = {
    0,
    199,
    409,
    839,
    1693,
    3391,
    10223
};

// setup 1 for 128KB shared memory per block
constexpr uint32_t SORT_ITEMS_PER_THREAD[] = {
    0,
    4,
    8,
    16,
    16,
    16,
    16
};

constexpr uint32_t SORT_BLOCK_SIZES[] = {
    0,
    32 * 2,
    32 * 2,
    32 * 2,
    32 * 4,
    32 * 7,
    32 * 20
};

#else

static_assert(false, "Unsupported shared memory size");

#endif


constexpr uint32_t DENSE_BIN_SIZES[] = {
    0, // The first bin for dense kernel is not used
    SHARED_MEM_PER_BLOCK[1] * 8 - 32,
    SHARED_MEM_PER_BLOCK[2] * 8 - 32,
    SHARED_MEM_PER_BLOCK[3] * 8 - 32,
    SHARED_MEM_PER_BLOCK[4] * 8 - 32,
    SHARED_MEM_PER_BLOCK[5] * 8 - 32,
    0  // the last bin for dense kernel is not used
};

// we're giving 1 extra byte per elements; 
// these are used for bitmap. 1 bit is enough 
constexpr uint32_t DENSE_NUMERIC_BIN_SIZES[] = {
    SHARED_MEM_PER_BLOCK[0] / (sizeof(data_t) * 2 + 1)  - 4,
    SHARED_MEM_PER_BLOCK[1] / (sizeof(data_t) + 1) - 4,
    SHARED_MEM_PER_BLOCK[2] / (sizeof(data_t) + 1) - 4,
    SHARED_MEM_PER_BLOCK[3] / (sizeof(data_t) + 1) - 4,
    SHARED_MEM_PER_BLOCK[4] / (sizeof(data_t) + 1) - 4,
    SHARED_MEM_PER_BLOCK[5] / (sizeof(data_t) + 1) - 4,
    0xFFFFFFFF
};


constexpr uint32_t HASH_NUMERIC_IDX_BITS[] = {
    0,
    32 - bits_needed(HASH_NUMERIC_BIN_SIZES[1]),
    32 - bits_needed(HASH_NUMERIC_BIN_SIZES[2]),
    32 - bits_needed(HASH_NUMERIC_BIN_SIZES[3]),
    32 - bits_needed(HASH_NUMERIC_BIN_SIZES[4]),
    32 - bits_needed(HASH_NUMERIC_BIN_SIZES[5]),
    32 - bits_needed(HASH_NUMERIC_BIN_SIZES[6])
};

// if changing these the template param need also be changed
constexpr uint32_t DENSE_NUMERIC_LARGEST_SMEM_SIZE = SHARED_MEM_PER_BLOCK[6];
constexpr uint32_t DENSE_NUMERIC_LARGEST_ITER_SIZE = DENSE_NUMERIC_LARGEST_SMEM_SIZE / (sizeof(data_t) + 2) - 4;
constexpr uint32_t DENSE_NUMERIC_LARGEST_ROW_OFFSET_SMEM = (DENSE_NUMERIC_LARGEST_SMEM_SIZE - (DENSE_NUMERIC_LARGEST_ITER_SIZE + 4) * (sizeof(data_t)) - (DENSE_NUMERIC_LARGEST_ITER_SIZE + 31) / 8 - 1024 / 32 * sizeof (uint32_t)) / sizeof(index_t);


constexpr uint32_t HASH_NUMERIC_USPARSE_UPPER_BOUNDS[] = {
    (SHARED_MEM_PER_BLOCK[1] - 256) / (sizeof(index_t) + sizeof(data_t)) / 4,
    (SHARED_MEM_PER_BLOCK[1] - 256) / (sizeof(index_t) + sizeof(data_t)) / 2
};


}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} \

#define CHECK_CUDA_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} \


