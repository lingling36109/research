#pragma once
#include "Common.h"
#include <limits>
#include <cuda_runtime.h>
#include <cstdio>


namespace somespgemm{
__host__ __device__ __forceinline__ uint32_t hashKernel(uint32_t id) {
	return id * 11;
}

// Hashmap itself would not check for overflow
template <int SIZE>
struct NumericHashmap {
  public:
    index_t *ids;
    data_t *values;
    index_t *occupancy;

    __device__ __forceinline__  void init(bool mainThread) {
        for (int i = threadIdx.x; i < SIZE; i += blockDim.x)
            ids[i] = HASH_UNUSED;
        for (int i = threadIdx.x; i < SIZE; i += blockDim.x)
            values[i] = 0;
        if (mainThread)  *occupancy = 0;
        __syncthreads();
    }

    __device__ __forceinline__ data_t* operator[](index_t id) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % SIZE;

        do {
            auto entry = ids[map_id];
            if (entry == id)
                return values + map_id;

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd(occupancy, 1);
                    }
                    return values + map_id;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= SIZE) map_id = 0;

        }  while (true);
    }

    __device__ __forceinline__ void atomicAddAt(index_t id, data_t val) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % SIZE;

        do {
            auto entry = ids[map_id];
            if (entry == id) {
                atomicAdd(values + map_id, val);
                return;
            }

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd(occupancy, 1);
                    }
                    atomicAdd(values + map_id, val);
                    return;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= SIZE) map_id = 0;

        }  while (true);
    }


    __device__ __forceinline__ bool nearFull() const { return *occupancy >= SIZE * HLL_HASHMAP_MAX_LOAD_FACTOR; }
};

template <int SIZE>
struct WarpNumericHashmap {
  public:
    index_t *ids;
    data_t *values;
    index_t *occupancy;

    __device__ __forceinline__  void init(bool mainThread, int tid, int group_size) {
        for (int i = tid; i < SIZE; i += group_size)
            ids[i] = HASH_UNUSED;
        for (int i = tid; i < SIZE; i += group_size)
            values[i] = 0;
        if (mainThread)  *occupancy = 0;
        __syncwarp();
    }

    __device__ __forceinline__ data_t* operator[](index_t id) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % SIZE;

        do {
            auto entry = ids[map_id];
            if (entry == id)
                return values + map_id;

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd(occupancy, 1);
                    }
                    return values + map_id;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= SIZE) map_id = 0;
        }  while (true);
    }
};

struct SymbolicHashmap {
  public:
    index_t *ids_shared;
    index_t *ids_global;
    index_t *occupancy;
    index_t capacity_shared;
    index_t capacity_global;


    __device__ __forceinline__  void init(bool mainThread) {
        for (int i = threadIdx.x; i < capacity_shared; i += blockDim.x)
            ids_shared[i] = HASH_UNUSED;
        if (mainThread)  *occupancy = 0;
        __syncthreads();
    }

    __device__ __forceinline__ bool operator[](index_t id) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % capacity_shared;

        do {
            auto entry = ids_shared[map_id];
            if (entry == id)
                return true;

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids_shared + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd(occupancy, 1);
                    }
                    return true;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= capacity_shared) map_id = 0;

        }  while (true);
    }

    __device__ __forceinline__ void global (index_t id) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % capacity_global;

        do {
            auto entry = ids_global[map_id];
            if (entry == id)
                return;

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids_global + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd(occupancy, 1);
                    }
                    return;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= capacity_global) map_id = 0;

        }  while (true);
    }
};

struct WarpSymbolicHashmap {
  public:
    index_t *ids;
    index_t *occupancy;
    index_t capacity;

    __device__ __forceinline__  void init(bool mainThread) {
        for (int i = threadIdx.x % WARP_SIZE; i < capacity; i += WARP_SIZE)
            ids[i] = HASH_UNUSED;
        if (mainThread)  *occupancy = 0;
        __syncwarp();
    }

    __device__ __forceinline__ bool operator[](index_t id) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % capacity;

        do {
            auto entry = ids[map_id];
            if (entry == id)
                return true;

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd_block(occupancy, 1);
                    }
                    return true;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= capacity) map_id = 0;
        }  while (true);
    }
};

// TODO: merge all Symbolic Hashmaps
struct FlexSymbolicHashmap {
  public:
    index_t *ids;
    index_t *occupancy;

    // int reserved;
    uint32_t capacity;

    __device__ __forceinline__  void init() {
        for (int i = threadIdx.x; i < capacity; i += blockDim.x)
            ids[i] = HASH_UNUSED;
        __syncthreads();
    }

    __device__ __forceinline__ bool operator[](index_t id) {
        index_t hashed_id = hashKernel(id); // can change to murmurhash
        index_t map_id = hashed_id % capacity;

        do {
            auto entry = ids[map_id];
            if (entry == id)
                return true;

            if (entry == HASH_UNUSED) {
                auto old_id = atomicCAS(ids + map_id, HASH_UNUSED, id);

                if (old_id == HASH_UNUSED || old_id == id) {
                    if (old_id == HASH_UNUSED) {
                        atomicAdd(occupancy, 1);
                    }
                    return true;
                }
            }
            map_id = (map_id + 1);
            if (map_id >= capacity) map_id = 0;
        }  while (true);
    }

    __device__ __forceinline__ void init_with_hashmap(index_t* old_ids, index_t old_size) {
        for (int i = threadIdx.x; i < old_size; i += blockDim.x) {
            index_t id = old_ids[i];
            if (id != HASH_UNUSED) {
                (*this)[id];
            }
        }
    }

};


}