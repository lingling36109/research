#pragma once
#include "AccumulatorCommon.cuh"
#include "Common.h"
#include "Hashmap.cuh"

namespace somespgemm {


// The input global Hashmap should be all 0xFF
// Overflow is possible only when we use assisted symbolic step
template<int LOG_NTHR, int HASH_SIZE, bool CHECK_OVERFLOW=false>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void hashSymbolicKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *num_products, 
    index_t *max_elements_b,
    index_t *my_row_ids,
    index_t *out_nnz_per_row,
    index_t *global_hashmap_buffer,
    uint32_t *global_buffer_bitmap,
    int num_global_buffers,
    size_t global_buffer_size
) {
    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    SymbolicHashmap hmap;
    hmap.capacity_shared = HASH_SIZE;
    hmap.capacity_global = 0;
    hmap.occupancy = (index_t*)(&dynamicShared[0]);
    hmap.ids_shared = (index_t *)(&dynamicShared[8 * sizeof(index_t)]);
    hmap.ids_global = nullptr;

    bool switched_to_gmem = false;
    index_t* global_map_idx = (index_t*)(&dynamicShared[4 * sizeof(index_t)]);

    index_t my_row;
    if (my_row_ids == nullptr) {
        my_row = blockIdx.x;
    } else {
        my_row = my_row_ids[blockIdx.x];
    }

    hmap.init(tid == 0);
    if (threadIdx.x < 4) {
        ((index_t*)dynamicShared)[tid] = 0;
    }
    if (threadIdx.x < 4) {
        global_map_idx[tid] = (index_t)(-1);
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

        if constexpr (CHECK_OVERFLOW) {
            bool need_switch = (!switched_to_gmem) && (*hmap.occupancy >= HASH_SIZE - (b_end - b_start));
            if (__builtin_expect(need_switch, 0)) {
                
                auto buffer_idx = getUniqueGlobalBuffer(
                    global_map_idx,
                    global_buffer_bitmap,
                    num_global_buffers
                );
                hmap.ids_global = &global_hashmap_buffer[buffer_idx * global_buffer_size]; // switch to global memory hashmap
                hmap.occupancy += 1; // the new occupancy is at dynamicShared[sizeof(index_t)]
                hmap.capacity_global = num_products[my_row] + (5 - num_products[my_row] % 6); // set to max necessary capacity. For global remember to make the sizes a little bit larger
                switched_to_gmem = true;
            }
        }

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            index_t b_col = matB_col_ind[b_element];

            if constexpr (CHECK_OVERFLOW) {
                if (__builtin_expect(switched_to_gmem, 0)) {
                    hmap.global(b_col);
                } else {
                    hmap[b_col];
                }
            } else {
                hmap[b_col];
            }
        }
        //__syncwarp();
    }

    __syncthreads();

    if (CHECK_OVERFLOW) {
        if (!switched_to_gmem) {
            index_t global_buffer_id = global_map_idx[0];
            if (global_buffer_id != (index_t)(-1)) {
                // switch to gmem
                hmap.ids_global = &global_hashmap_buffer[global_buffer_id * global_buffer_size];
                hmap.occupancy += 1;
                hmap.capacity_global = num_products[my_row] + (5 - num_products[my_row] % 6);
                switched_to_gmem = true;
            }
        }
        __syncthreads();

        if (switched_to_gmem) {
            // merge shared memory hashmap to global memory
            index_t* shared_map = (index_t*)(dynamicShared + 8 * sizeof(index_t));
            for (int i = tid; i < HASH_SIZE; i += blockDim.x) {
                index_t idx = shared_map[i];
                if (idx != HASH_UNUSED) {
                    hmap.global(idx);
                }
            }
            __syncthreads();

            // clean the global hashmap
            for (int i = tid; i < hmap.capacity_global; i += blockDim.x) {
                hmap.ids_global[i] = HASH_UNUSED;
            }

            __syncthreads();
            // release the global buffer
            if (tid == 0) {
                releaseGlobalBuffer(global_buffer_bitmap, global_map_idx[0]);
            }
        }
    }

    if (tid == 0) {
        out_nnz_per_row[my_row] = *hmap.occupancy;
    }
}


// One block is responsible for two rows in parallel
// Currently *not* supporting more than two rows
template<int LOG_NTHR, int HASH_SIZE, bool CHECK_OVERFLOW=false>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void hashSymbolicSubWarpKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *num_products, 
    index_t *max_elements_b,
    index_t *my_row_ids,
    index_t *out_nnz_per_row,
    index_t num_rows,
    index_t *global_hashmap_buffer,
    uint32_t *global_buffer_bitmap,
    int num_global_buffers,
    size_t global_buffer_size
) {

    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x % WARP_SIZE;
    index_t wid = threadIdx.x / WARP_SIZE;

    WarpSymbolicHashmap hmap;

    hmap.occupancy = (index_t *)(&dynamicShared[sizeof(index_t) * (wid*2)]);
    hmap.ids = (index_t *)(&dynamicShared[8 * sizeof(index_t) + wid * HASH_SIZE * sizeof(index_t)]);
    hmap.capacity = HASH_SIZE;

    bool switched_to_gmem = false;
    index_t* global_map_idx = (index_t*)(&dynamicShared[4 * sizeof(index_t) + wid * 2 * sizeof(index_t)]);

    if (threadIdx.x < 4) {
        hmap.occupancy[tid] = 0;
        global_map_idx[tid] = (index_t)(-1);
    }
    __syncthreads();

    if (blockIdx.x * 2 + wid >= num_rows) {
        return;
    }
    index_t my_row;
    if (my_row_ids == nullptr) {
        my_row = blockIdx.x * 2 + wid;
    } else {
        my_row = my_row_ids[blockIdx.x * 2 + wid];
    }

    hmap.init(tid == 0);
    __syncwarp();

    index_t start_element_a = matA_row_offsets[my_row];
    index_t end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];
    
    index_t log_nthr_per_a_elem = localLoadBalance(
        end_element_a - start_element_a,
        num_products[my_row],
        max_elements_b[my_row],
        1, 5
    );

    index_t my_group_id = tid >> log_nthr_per_a_elem;
    index_t num_groups = WARP_SIZE >> log_nthr_per_a_elem;
    index_t group_size = 1 << log_nthr_per_a_elem;
    index_t my_id_in_group = tid & ((1 << log_nthr_per_a_elem) - 1);

    for (index_t element_a = start_element_a + my_group_id;
         element_a < end_element_a;
         element_a += num_groups) {

        index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
        index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            index_t b_col = matB_col_ind[b_element];

            if (CHECK_OVERFLOW) {
                if (!switched_to_gmem && (*hmap.occupancy >= HASH_SIZE - WARP_SIZE * HASH_SYMBOLIC_GMEM_FULL_COE)) {
                    auto buffer_idx = getUniqueGlobalBuffer(
                        global_map_idx,
                        global_buffer_bitmap,
                        num_global_buffers
                    );
                    hmap.ids = &global_hashmap_buffer[buffer_idx * global_buffer_size]; // switch to global memory hashmap
                    hmap.occupancy += 1; // the new occupancy is at dynamicShared[sizeof(index_t)]
                    hmap.capacity = num_products[my_row] + (5 - num_products[my_row] % 6); // set to max necessary capacity. For global remember to make the sizes a little bit larger
                    switched_to_gmem = true;
                }
            }

            hmap[b_col];
        }
        //__syncwarp();
    }

    __syncwarp();

    if (CHECK_OVERFLOW) {
        if (!switched_to_gmem) {
            index_t global_buffer_id = global_map_idx[0];
            if (global_buffer_id != (index_t)(-1)) {
                // switch to gmem
                hmap.ids = &global_hashmap_buffer[global_buffer_id * global_buffer_size]; 
                hmap.occupancy += 1;
                hmap.capacity = num_products[my_row] + (5 - num_products[my_row] % 6);
                switched_to_gmem = true;
            }
        }

        __syncwarp();

        if (switched_to_gmem) {
            // merge shared memory hashmap to global memory
            index_t* shared_map = (index_t*)(dynamicShared + 8 * sizeof(index_t) + wid * HASH_SIZE * sizeof(index_t));
            for (int i = tid; i < HASH_SIZE; i += WARP_SIZE) {
                index_t idx = shared_map[i];
                if (idx != HASH_UNUSED) {
                    hmap[idx];
                }
            }
            __syncwarp();

            // clean the global hashmap
            for (int i = tid; i < hmap.capacity; i += WARP_SIZE) {
                hmap.ids[i] = HASH_UNUSED;
            }

            __syncwarp();
            // release the global buffer
            if (tid == 0) {
                releaseGlobalBuffer(global_buffer_bitmap, global_map_idx[0]);
            }
        }
    }

    if (tid == 0) {
        out_nnz_per_row[my_row] = *hmap.occupancy;
    }
}

// accumulates in global memory
template<int LOG_NTHR>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void hashSymbolicGMEMKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *num_products, 
    index_t *max_elements_b,
    index_t *my_row_ids,
    index_t *out_nnz_per_row,
    index_t *buffer_global_hashmap,
    index_t my_hash_size_shared,
    index_t my_hash_size_global, 
    index_t num_rows,
    double global_compaction,
    bool always_use_gmem
    ) {
    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    FlexSymbolicHashmap hmap;

    hmap.occupancy = (index_t *)(&dynamicShared[0]);
    bool switched_to_gmem = false;

    index_t iter = blockIdx.x;
    while (iter < num_rows) {
        index_t my_row;
        if (my_row_ids == nullptr) {
            my_row = iter;
        } else {
            my_row = my_row_ids[iter];
        }

        // init hashmap
        __syncthreads();
        if (switched_to_gmem) {
            for (int i = tid; i < my_hash_size_global; i += blockDim.x) {
                hmap.ids[i] = HASH_UNUSED;
            }
        }

        bool start_with_gmem;

        // determine whether to use global memory based on the sample results
        if (num_products[my_row] / global_compaction >= my_hash_size_shared * HASH_SYMBOLIC_GMEM_INIT_SWITCH_COE || always_use_gmem) {
            switched_to_gmem = true;
            start_with_gmem = true;
            hmap.occupancy = (index_t *)(&dynamicShared[sizeof(index_t)]);
            hmap.ids = &buffer_global_hashmap[blockIdx.x * my_hash_size_global]; // switch to global memory hashmap
            hmap.capacity = my_hash_size_global;
        } else {
            switched_to_gmem = false;
            start_with_gmem = false;
            hmap.capacity = my_hash_size_shared;
            hmap.occupancy = (index_t *)(&dynamicShared[0]);
            hmap.ids = (index_t *)(&dynamicShared[sizeof(index_t) * 4]);
            hmap.init();
        }

        if (tid == 0) {
            for (int i = 0; i < 4; i++) {
                ((index_t*)dynamicShared)[i] = 0;
            }
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

            for (index_t b_element = b_start + my_id_in_group;
                b_element < b_end;
                b_element += group_size) {

                // CHECK OVERFLOW
                if ( !switched_to_gmem && (*hmap.occupancy >= hmap.capacity - blockDim.x * HASH_SYMBOLIC_GMEM_FULL_COE))  {
                    hmap.ids = &buffer_global_hashmap[blockIdx.x * my_hash_size_global]; // switch to global memory hashmap
                    hmap.capacity = my_hash_size_global;
                    hmap.occupancy = (index_t *)(&dynamicShared[sizeof(index_t)]);
                    switched_to_gmem = true;
                }

                index_t b_col = matB_col_ind[b_element];
                hmap[b_col];
            }
            //__syncwarp();
        }

        __syncthreads();

        // ensure all threads are switched to gmem if it happens
        if (!switched_to_gmem) {
            index_t value = *(index_t*)(&dynamicShared[sizeof(index_t)]);
            if (value != 0) {
                switched_to_gmem = true;
                hmap.ids = &buffer_global_hashmap[blockIdx.x * my_hash_size_global]; 
                hmap.capacity = my_hash_size_global;
                hmap.occupancy = (index_t *)(&dynamicShared[sizeof(index_t)]);
            }
        }

        // if switched to gmem, we need to merge the shared memory hashmap to global memory
        if ((!start_with_gmem) && switched_to_gmem) {
            index_t* shared_map = (index_t*)(dynamicShared + sizeof(index_t) * 4);
            for (int i = tid; i < my_hash_size_shared; i += blockDim.x) {
                index_t idx = shared_map[i];
                if (idx != HASH_UNUSED) {
                    hmap[idx];
                }
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            out_nnz_per_row[my_row] = *hmap.occupancy;
        }

        iter += gridDim.x;
    }
}


// this is using speck's load balancing style.
// compact and sort output of hash accumulator
template<int SIZE>
__device__ __forceinline__ void compactAndSort(index_t* counter, index_t* idx, data_t* val, index_t* idx_out, data_t* val_out) {
    if (threadIdx.x == 0) {
        *counter = 0;
    }
    __syncthreads();

    // Step 1 Compact
    auto total_iter = (SIZE + blockDim.x - 1) / blockDim.x;

    #pragma unroll
    for (int it = 0; it < total_iter; it++) {
        int i = it * blockDim.x + threadIdx.x;
        data_t my_val;
        index_t my_idx;
        if (i < SIZE) {
            my_val = val[i];
            my_idx = idx[i];
        } else {
            my_val = 0;
            my_idx = HASH_UNUSED;
        }
        __syncthreads();    // this syncthreads force all threads to loop together

        if (my_idx != HASH_UNUSED) {
            index_t pos = atomicAdd(counter, 1);
            idx[pos] = my_idx;
            val[pos] = my_val;
        }
    }

    __syncthreads();
    // Step 2 Sort
    index_t count = *counter;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        auto current_idx = idx[i];
        int loc = 0;
        for (int j = 0; j < count; j++) {
            if(idx[j] < current_idx) {
                loc++;
            }
        }
        idx_out[loc] = current_idx;
        val_out[loc] = val[i];
    }
}


// for long rows we compact only and do not sort
template<int SIZE>
__device__ __forceinline__ void compactOnly(index_t* counter, index_t* idx, data_t* val, index_t* idx_out, data_t* val_out) {
    if (threadIdx.x == 0) {
        *counter = 0;
    }
    __syncthreads();

    // Step 1 Compact
    #pragma unroll
    for (int i = threadIdx.x; i < SIZE; i += blockDim.x) {
        auto my_val = val[i];
        auto my_idx = idx[i];
        // __syncthreads();
        if (my_idx != HASH_UNUSED) {
            index_t pos = atomicAdd(counter, 1);
            idx_out[pos] = my_idx;
            val_out[pos] = my_val;
        }
    }
    __syncthreads();
}

template<int LOG_NTHR, int HASH_SIZE, bool CHECK_OVERFLOW=false, bool HYBRID_HASHMAP=false, typename offset_t=index_t>
__global__ __launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR) void hashNumericKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *num_products, 
    index_t *max_elements_b,
    index_t *inout_my_row_ids,
    offset_t *output_row_offsets, index_t *out_cols, data_t *out_vals,     // output
    index_t *out_nnz_per_row=nullptr,
    index_t *out_num_overflow_rows=nullptr, 
    index_t *out_overflow_row_ids=nullptr, 
    index_t *out_overflow_buffer_allocated=nullptr,
    data_t *global_hashmap_buffer=nullptr,
    uint32_t *global_buffer_bitmap=nullptr,
    int num_global_buffers=0,
    size_t global_buffer_size=0
) {
    __shared__ index_t global_buffer_idx;

    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x;

    NumericHashmap<HASH_SIZE> hmap;

    hmap.occupancy = (index_t *)(&dynamicShared[0]);
    hmap.ids = (index_t *)(&dynamicShared[4 * sizeof(index_t)]);

    if constexpr (!HYBRID_HASHMAP) {
        hmap.values = (data_t *)(&dynamicShared[4 * sizeof(index_t) + sizeof(index_t) * ((HASH_SIZE +1) / 2 * 2)]);
    } else {
        if (threadIdx.x == 0) {
            global_buffer_idx = (index_t)(-1);
            
            auto buffer_idx = getUniqueGlobalBuffer(
                &global_buffer_idx,
                global_buffer_bitmap,
                num_global_buffers
            );
        }
        __syncthreads();
        hmap.values = global_hashmap_buffer + global_buffer_idx * global_buffer_size;
    }

    index_t my_row;
    if (inout_my_row_ids == nullptr) {
        my_row = blockIdx.x;
    } else {
        my_row = inout_my_row_ids[blockIdx.x];
    }

    hmap.init(tid == 0);
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

        data_t a_value = matA_values[element_a];
        index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
        index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];

        for (index_t b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            data_t b_value =  matB_values[b_element];
            index_t b_col = matB_col_ind[b_element];
            
            if (CHECK_OVERFLOW) {
                if (hmap.nearFull()) {
                    break;
                }
            }
            auto loc = hmap[b_col];
            atomicAdd(loc, a_value * b_value);
        }
        //__syncwarp();
    }

    __syncthreads();

    if (CHECK_OVERFLOW) {
        if (hmap.nearFull()) {
            if (tid == 0) {
                auto loc = atomicAdd(out_num_overflow_rows, 1);
                out_overflow_row_ids[loc] = my_row;

                index_t add_num = num_products[my_row];
                if (add_num > num_cols_b) add_num = num_cols_b;
                auto buffer_loc = atomicAdd(out_overflow_buffer_allocated, add_num);
                output_row_offsets[my_row] = buffer_loc;
                inout_my_row_ids[blockIdx.x] = (index_t)(-1);

                if constexpr (HYBRID_HASHMAP) {
                    releaseGlobalBuffer(global_buffer_bitmap, global_buffer_idx);
                }
            }
            return; // we stop processing if the hashmap is full
        }
    }
    
    // put results back to global memory
    if (*hmap.occupancy <= NUM_HASH_INPLACE_SORT_THRESHOLD) {
        compactAndSort<HASH_SIZE>((index_t*)(&dynamicShared[sizeof(index_t)]), 
            hmap.ids, hmap.values,
            &out_cols[output_row_offsets[my_row]],
            &out_vals[output_row_offsets[my_row]]);
    } else {
        compactOnly<HASH_SIZE>((index_t*)(&dynamicShared[sizeof(index_t)]), 
            hmap.ids, hmap.values,
            &out_cols[output_row_offsets[my_row]],
            &out_vals[output_row_offsets[my_row]]);
    }

    if (out_nnz_per_row != nullptr) {
        if (tid == 0) {
            out_nnz_per_row[my_row] = *hmap.occupancy;
        }
    }

    if constexpr (HYBRID_HASHMAP) {
        // release the global buffer
        if (tid == 0) {
            releaseGlobalBuffer(global_buffer_bitmap, global_buffer_idx);
        }
    }
    return;
}

// this kernel would be responsible for multiple rows per block of 64 threads
// While it should support maximum of 16 concurrent rows, it seems at most 4 is supported currently
template<int LOG_NTHR_PER_ROW, int HASH_SIZE>
__global__ __launch_bounds__(64, 32) void hashNumericSubWarpKernel(
    index_t *matA_row_offsets, index_t *matA_col_ind, data_t *matA_values, index_t num_rows_a, index_t num_cols_a, index_t nnz_a, 
    index_t *matB_row_offsets, index_t *matB_col_ind, data_t *matB_values, index_t num_rows_b, index_t num_cols_b, index_t nnz_b,
    index_t *num_products, 
    index_t *inout_my_row_ids,
    size_t *output_row_offsets, index_t *out_cols, data_t *out_vals,
    index_t *out_nnz_per_row=nullptr,
    index_t process_upper_bound = 0xFFFFFFFF
) {
    constexpr int NUM_ROWS = 64 >> LOG_NTHR_PER_ROW;
    constexpr int HASH_SIZE_ROUNDED = HASH_SIZE / 2 * 2 + 2;

    extern __shared__ uint8_t dynamicShared[];
    index_t tid = threadIdx.x % (1 << LOG_NTHR_PER_ROW);
    index_t row_group_id = threadIdx.x >> LOG_NTHR_PER_ROW;
    
    WarpNumericHashmap<HASH_SIZE> hmap;

    hmap.occupancy = (index_t*)(&dynamicShared[0]) + row_group_id;
    hmap.ids = (index_t*)(&dynamicShared[32 * sizeof(index_t) + (sizeof(index_t) + sizeof(data_t)) * HASH_SIZE_ROUNDED * row_group_id]);
    hmap.values = (data_t*)(hmap.ids + HASH_SIZE_ROUNDED);
    index_t* row_ids = (index_t*)(&dynamicShared[0]) + 16; 

    index_t my_row;
    index_t my_products = 1024;

    if (blockIdx.x * NUM_ROWS + row_group_id < num_rows_a) {
        if (inout_my_row_ids == nullptr) {
            my_row = blockIdx.x * NUM_ROWS + row_group_id;
        } else {
            my_row = inout_my_row_ids[blockIdx.x * NUM_ROWS + row_group_id];
        }
        my_products = num_products[my_row];
    }

    hmap.init(tid == 0, tid, (1 << LOG_NTHR_PER_ROW));

    if (my_products > process_upper_bound) {
        row_ids[row_group_id] = (index_t)(-1);
    } else {
        row_ids[row_group_id] = my_row;
        
        index_t start_element_a = matA_row_offsets[my_row];
        index_t end_element_a = (my_row+1 == num_rows_a) ? nnz_a: matA_row_offsets[my_row+1];
        
        constexpr int log_nthr_per_a_elem = 2;    // fixed at 4 threads per a element

        index_t my_group_id = tid >> log_nthr_per_a_elem;
        index_t num_groups = 1 << (LOG_NTHR_PER_ROW - log_nthr_per_a_elem);
        index_t my_id_in_group = tid & ((1 << log_nthr_per_a_elem) - 1);
        index_t group_size = 1 << log_nthr_per_a_elem;

        for (index_t element_a = start_element_a + my_group_id;
            element_a < end_element_a;
            element_a += num_groups) {

            data_t a_value = matA_values[element_a];
            index_t b_start = matB_row_offsets[matA_col_ind[element_a]];
            index_t b_end = matB_row_offsets[matA_col_ind[element_a]+1];

            for (index_t b_element = b_start + my_id_in_group;
                b_element < b_end;
                b_element += group_size) {

                data_t b_value = matB_values[b_element];
                index_t b_col = matB_col_ind[b_element];
                
                auto loc = hmap[b_col];
                atomicAdd(loc, a_value * b_value);
            }
        }
    }


    __syncthreads();
    if (out_nnz_per_row != nullptr && my_products <= process_upper_bound) {
        if (tid == 0) {
            out_nnz_per_row[my_row] = *hmap.occupancy;
        }
    }
    __syncthreads();
    
    // put results back to global memory. Here all threads cooperate
    for (int i = 0; i < NUM_ROWS; i++) {
        index_t* ids = (index_t*)(&dynamicShared[32 * sizeof(index_t) + (sizeof(index_t) + sizeof(data_t)) * HASH_SIZE_ROUNDED * i]);
        data_t* vals = (data_t*)(ids + HASH_SIZE_ROUNDED);
        index_t cur_row = row_ids[i];
        if (cur_row == (index_t)(-1)) {
            continue;
        }
        compactAndSort<HASH_SIZE>(
            (index_t*)(&dynamicShared[0]), 
            ids, vals,
            &out_cols[output_row_offsets[cur_row]],
            &out_vals[output_row_offsets[cur_row]]
        );
    }

    return;
}

}
