#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "Common.h"
#include "Utils.h"

namespace somespgemm {

// Spy-plot image segmentation for SpGEMM:
// 1. Build a downsampled density map of A's nonzero pattern (grid_h x grid_w tiles).
// 2. Threshold + initialize labels (hot tiles = candidate dense block seeds).
// 3. Run iterative GPU connected-components on the 4-connected tile grid.
// 4. Pick a per-grid-row dominant label, RLE on host to build row-segments.
// 5. Per-segment stats reduction + classification kernel.

struct DensitySqOp {
    __host__ __device__ uint64_t operator()(uint32_t v) const {
        return (uint64_t)v * (uint64_t)v;
    }
};

struct DensityCastOp {
    __host__ __device__ uint64_t operator()(uint32_t v) const {
        return (uint64_t)v;
    }
};

enum SegmentClass : uint32_t {
    SEG_SPARSE_UNIFORM = 0,
    SEG_DENSE          = 1,
    SEG_POWER_LAW      = 2,
    SEG_BANDED         = 3,
    SEG_CLASS_COUNT    = 4
};

struct SegmentRecord {
    uint32_t row_start;
    uint32_t row_end;
    uint32_t col_left;
    uint32_t col_right;
    uint32_t cls;            // SegmentClass
    uint32_t row_count;
    uint64_t sum_products;
    uint64_t sum_a_nnz;
    uint32_t max_b_row_len;
    uint32_t max_a_row_len;
    double   mean_row_len;
    double   var_row_len;    // population variance of per-row product count
    double   local_er;       // sum_products / sum_a_nnz
};

struct SegmentationResult {
    std::vector<SegmentRecord> segments;
    uint32_t grid_h = 0;
    uint32_t grid_w = 0;
    uint32_t tile_rows = 0;   // rows per tile
    uint32_t tile_cols = 0;   // cols per tile
    uint32_t num_components = 0;
    double   density_mean = 0.0;
    double   density_stddev = 0.0;
    float    time_density_ms = 0.0f;
    float    time_threshold_ms = 0.0f;
    float    time_cc_ms = 0.0f;
    float    time_classify_ms = 0.0f;
    int      cc_iterations = 0;
};

// ------- Kernel 0: density map -------
// One warp per A-row. Each lane walks a strided slice of the row's nonzeros
// and atomicAdds into the (row/T_r, col/T_c) tile counter.
__global__ void densityMapKernel(
    const index_t* __restrict__ A_row_offsets,
    const index_t* __restrict__ A_col_ids,
    size_t A_rows,
    size_t A_nnz,
    uint32_t grid_h,
    uint32_t grid_w,
    uint32_t tile_rows,
    uint32_t tile_cols,
    uint32_t* __restrict__ d_density
) {
    size_t warp_id_global = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id_global >= A_rows) return;

    size_t row = warp_id_global;
    index_t row_st = A_row_offsets[row];
    index_t row_ed = (row + 1 == A_rows) ? (index_t)A_nnz : A_row_offsets[row + 1];

    uint32_t tr = (uint32_t)(row / tile_rows);
    if (tr >= grid_h) tr = grid_h - 1;

    for (index_t j = row_st + lane; j < row_ed; j += WARP_SIZE) {
        index_t c = A_col_ids[j];
        uint32_t tc = (uint32_t)(c / tile_cols);
        if (tc >= grid_w) tc = grid_w - 1;
        atomicAdd(&d_density[(size_t)tr * grid_w + tc], 1u);
    }
}

// ------- Kernel a: threshold + label init -------
// One thread per tile. Hot tile -> label = its flat index; cold -> 0xFFFFFFFF.
__global__ void thresholdAndInitLabelsKernel(
    const uint32_t* __restrict__ d_density,
    uint32_t* __restrict__ d_labels,
    uint32_t num_tiles,
    double mean,
    double stddev,
    double k
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tiles) return;
    double thresh = mean + k * stddev;
    if ((double)d_density[tid] >= thresh && d_density[tid] > 0u) {
        d_labels[tid] = tid;
    } else {
        d_labels[tid] = 0xFFFFFFFFu;
    }
}

// ------- Kernel b1: CC propagation iteration -------
// Each thread looks at its 4 neighbors. If self is hot and any hot neighbor has
// a smaller label, take the min. Sets d_changed if any update occurs.
__global__ void ccPropagationKernel(
    uint32_t* __restrict__ d_labels,
    uint32_t grid_h,
    uint32_t grid_w,
    int* __restrict__ d_changed
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= grid_w || y >= grid_h) return;
    uint32_t idx = y * grid_w + x;
    uint32_t my_label = d_labels[idx];
    if (my_label == 0xFFFFFFFFu) return;

    uint32_t best = my_label;
    if (x > 0) {
        uint32_t l = d_labels[idx - 1];
        if (l != 0xFFFFFFFFu && l < best) best = l;
    }
    if (x + 1 < grid_w) {
        uint32_t l = d_labels[idx + 1];
        if (l != 0xFFFFFFFFu && l < best) best = l;
    }
    if (y > 0) {
        uint32_t l = d_labels[idx - grid_w];
        if (l != 0xFFFFFFFFu && l < best) best = l;
    }
    if (y + 1 < grid_h) {
        uint32_t l = d_labels[idx + grid_w];
        if (l != 0xFFFFFFFFu && l < best) best = l;
    }
    if (best < my_label) {
        d_labels[idx] = best;
        atomicExch(d_changed, 1);
    }
}

// ------- Kernel b2: pointer jumping flatten -------
// Treats each label as a pointer to another tile; replaces with label[label]
// until labels[idx] == labels[labels[idx]]. Caps iterations to avoid runaway.
__global__ void ccPointerJumpKernel(
    uint32_t* __restrict__ d_labels,
    uint32_t num_tiles,
    int* __restrict__ d_changed
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tiles) return;
    uint32_t l = d_labels[tid];
    if (l == 0xFFFFFFFFu) return;
    uint32_t parent = d_labels[l];
    if (parent != l) {
        d_labels[tid] = parent;
        atomicExch(d_changed, 1);
    }
}

// ------- Kernel c: per-grid-row dominant label -------
// One block per grid row. Each thread inspects one tile of its row and
// votes into a small per-row label histogram in shared memory using
// linear-probing dictionary. Output: row_band_label[grid_row], row_band_count[grid_row].
// For correctness with simplicity, use a per-block reduction array of (label, count) pairs
// kept in shared mem; final write picks the max-count label.
//
// Limit: assumes grid_w <= 1024. With grid_max defaulting to 256 this holds.
__global__ void rowBandDominantLabelKernel(
    const uint32_t* __restrict__ d_labels,
    uint32_t grid_h,
    uint32_t grid_w,
    uint32_t* __restrict__ d_row_band_label,
    uint32_t* __restrict__ d_row_band_hot_count
) {
    extern __shared__ uint32_t smem[];
    // smem layout: [grid_w] keys followed by [grid_w] counts
    uint32_t* keys   = smem;
    uint32_t* counts = smem + grid_w;

    uint32_t row = blockIdx.x;
    if (row >= grid_h) return;

    __shared__ uint32_t s_hot_total;
    if (threadIdx.x == 0) s_hot_total = 0;

    // init
    for (uint32_t i = threadIdx.x; i < grid_w; i += blockDim.x) {
        keys[i]   = 0xFFFFFFFFu;
        counts[i] = 0;
    }
    __syncthreads();

    // tally
    for (uint32_t x = threadIdx.x; x < grid_w; x += blockDim.x) {
        uint32_t lbl = d_labels[row * grid_w + x];
        if (lbl == 0xFFFFFFFFu) continue;
        atomicAdd(&s_hot_total, 1u);
        // linear-probe insert/increment
        uint32_t h = (lbl * 2654435761u) % grid_w;
        for (uint32_t step = 0; step < grid_w; ++step) {
            uint32_t slot = (h + step) % grid_w;
            uint32_t cur  = atomicCAS(&keys[slot], 0xFFFFFFFFu, lbl);
            if (cur == 0xFFFFFFFFu || cur == lbl) {
                atomicAdd(&counts[slot], 1u);
                break;
            }
        }
    }
    __syncthreads();

    // pick max count
    if (threadIdx.x == 0) {
        uint32_t best_lbl = 0xFFFFFFFFu;
        uint32_t best_cnt = 0;
        for (uint32_t i = 0; i < grid_w; ++i) {
            if (counts[i] > best_cnt) {
                best_cnt = counts[i];
                best_lbl = keys[i];
            }
        }
        d_row_band_label[row]     = best_lbl;
        d_row_band_hot_count[row] = s_hot_total;
    }
}

// ------- Kernel d: per-segment reduce + classify -------
// One block per segment. Each block streams over its [row_start, row_end) range
// using all threads, reducing per-row stats into per-segment aggregates and
// computing classification thresholds on the fly.
__global__ void perSegmentReduceClassifyKernel(
    const uint32_t* __restrict__ d_seg_row_start,
    const uint32_t* __restrict__ d_seg_row_end,
    const index_t*  __restrict__ d_num_products,
    const index_t*  __restrict__ d_max_b_row_len,
    const index_t*  __restrict__ d_avg_b_row_len,
    const index_t*  __restrict__ d_b_col_idx_left,
    const index_t*  __restrict__ d_b_col_idx_right,
    const index_t*  __restrict__ d_A_row_offsets,
    size_t A_rows,
    size_t A_nnz,
    uint32_t B_cols,
    SegmentRecord* __restrict__ d_records,
    // classification thresholds:
    double dense_density_threshold,        // products per row / col_span
    double power_law_cv_threshold,         // coefficient of variation cutoff
    double banded_col_span_ratio,          // (col_right-col_left) / B_cols below this -> narrow
    int    estimation_input_expansion_threshold
) {
    int seg = blockIdx.x;
    uint32_t row_start = d_seg_row_start[seg];
    uint32_t row_end   = d_seg_row_end[seg];
    uint32_t row_count = row_end - row_start;

    // accumulators
    typedef cub::BlockReduce<uint64_t, 128> BlockReduceU64;
    typedef cub::BlockReduce<uint32_t, 128> BlockReduceU32;

    __shared__ typename BlockReduceU64::TempStorage temp_u64;
    __shared__ typename BlockReduceU32::TempStorage temp_u32;

    uint64_t local_sum_products = 0;
    uint64_t local_sum_a_nnz    = 0;
    uint64_t local_sum_sq_prod  = 0;
    uint32_t local_max_b_row    = 0;
    uint32_t local_max_a_row    = 0;
    uint32_t local_col_left     = 0xFFFFFFFFu;
    uint32_t local_col_right    = 0;

    for (uint32_t r = row_start + threadIdx.x; r < row_end; r += blockDim.x) {
        uint32_t p = d_num_products[r];
        local_sum_products += p;
        local_sum_sq_prod  += (uint64_t)p * (uint64_t)p;

        index_t a_st = d_A_row_offsets[r];
        index_t a_ed = (r + 1 == A_rows) ? (index_t)A_nnz : d_A_row_offsets[r + 1];
        uint32_t a_len = (uint32_t)(a_ed - a_st);
        local_sum_a_nnz += a_len;
        if (a_len > local_max_a_row) local_max_a_row = a_len;

        uint32_t mb = d_max_b_row_len[r];
        if (mb > local_max_b_row) local_max_b_row = mb;
        uint32_t cl = d_b_col_idx_left[r];
        uint32_t cr = d_b_col_idx_right[r];
        if (cl < local_col_left)  local_col_left  = cl;
        if (cr > local_col_right) local_col_right = cr;
    }

    // Reduce — only thread 0 receives valid results; sync between reuses.
    uint64_t sum_products = BlockReduceU64(temp_u64).Sum(local_sum_products);
    __syncthreads();
    uint64_t sum_a_nnz    = BlockReduceU64(temp_u64).Sum(local_sum_a_nnz);
    __syncthreads();
    uint64_t sum_sq_prod  = BlockReduceU64(temp_u64).Sum(local_sum_sq_prod);
    __syncthreads();
    uint32_t max_b_row    = BlockReduceU32(temp_u32).Reduce(local_max_b_row, cub::Max());
    __syncthreads();
    uint32_t max_a_row    = BlockReduceU32(temp_u32).Reduce(local_max_a_row, cub::Max());
    __syncthreads();
    uint32_t col_left     = BlockReduceU32(temp_u32).Reduce(local_col_left, cub::Min());
    __syncthreads();
    uint32_t col_right    = BlockReduceU32(temp_u32).Reduce(local_col_right, cub::Max());
    __syncthreads();

    if (threadIdx.x == 0) {
        SegmentRecord rec{};
        rec.row_start     = row_start;
        rec.row_end       = row_end;
        rec.col_left      = (col_left == 0xFFFFFFFFu) ? 0u : col_left;
        rec.col_right     = col_right;
        rec.row_count     = row_count;
        rec.sum_products  = sum_products;
        rec.sum_a_nnz     = sum_a_nnz;
        rec.max_b_row_len = max_b_row;
        rec.max_a_row_len = max_a_row;

        double mean_p = (row_count > 0) ? (double)sum_products / (double)row_count : 0.0;
        double var_p  = 0.0;
        if (row_count > 0) {
            double mean_sq = (double)sum_sq_prod / (double)row_count;
            var_p = mean_sq - mean_p * mean_p;
            if (var_p < 0.0) var_p = 0.0;
        }
        double cv     = (mean_p > 0.0) ? sqrt(var_p) / mean_p : 0.0;
        double er     = (sum_a_nnz > 0) ? (double)sum_products / (double)sum_a_nnz : 0.0;
        double col_span_ratio = (B_cols > 0)
            ? (double)(rec.col_right + 1 - rec.col_left) / (double)B_cols
            : 0.0;
        double row_density = (col_span_ratio > 0.0) ? mean_p / ((double)B_cols * col_span_ratio) : 0.0;

        rec.mean_row_len = mean_p;
        rec.var_row_len  = var_p;
        rec.local_er     = er;

        // classify (priority: BANDED > DENSE > POWER_LAW > SPARSE_UNIFORM)
        uint32_t cls = SEG_SPARSE_UNIFORM;
        if (col_span_ratio > 0.0 && col_span_ratio < banded_col_span_ratio
            && mean_p > 0.0 && er >= (double)estimation_input_expansion_threshold) {
            cls = SEG_BANDED;
        } else if (row_density >= dense_density_threshold) {
            cls = SEG_DENSE;
        } else if (cv >= power_law_cv_threshold) {
            cls = SEG_POWER_LAW;
        }
        rec.cls = cls;

        d_records[seg] = rec;
    }
}

// ------- Host driver -------
inline void runSegmentation(
    const index_t* d_A_row_offsets,
    const index_t* d_A_col_ids,
    size_t A_rows,
    size_t A_nnz,
    size_t B_cols,
    const index_t* d_num_products,
    const index_t* d_max_b_row_len,
    const index_t* d_avg_b_row_len,
    const index_t* d_b_col_idx_left,
    const index_t* d_b_col_idx_right,
    const Config& config,
    cudaStream_t stream,
    SegmentationResult& out
) {
    if (A_rows == 0 || B_cols == 0) {
        out.segments.clear();
        return;
    }

    // grid dimensions
    uint32_t grid_max = (uint32_t)std::max(8, config.segmentation_grid_max);
    uint32_t grid_h = (uint32_t)std::min<size_t>(grid_max, A_rows);
    uint32_t grid_w = (uint32_t)std::min<size_t>(grid_max, B_cols);
    if (grid_h == 0) grid_h = 1;
    if (grid_w == 0) grid_w = 1;
    uint32_t tile_rows = (uint32_t)((A_rows + grid_h - 1) / grid_h);
    uint32_t tile_cols = (uint32_t)((B_cols + grid_w - 1) / grid_w);
    uint32_t num_tiles = grid_h * grid_w;

    out.grid_h    = grid_h;
    out.grid_w    = grid_w;
    out.tile_rows = tile_rows;
    out.tile_cols = tile_cols;

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);

    // ------- Stage 1: density map -------
    uint32_t* d_density = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_density, sizeof(uint32_t) * num_tiles, stream));
    CHECK_CUDA(cudaMemsetAsync(d_density, 0, sizeof(uint32_t) * num_tiles, stream));

    cudaEventRecord(e0, stream);
    {
        constexpr int BS = 128;
        constexpr int WARPS_PER_BLOCK = BS / WARP_SIZE;
        size_t nblocks = (A_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        // cap at something sane to avoid massive grids; CC of warps still scales
        if (nblocks == 0) nblocks = 1;
        densityMapKernel<<<(unsigned int)nblocks, BS, 0, stream>>>(
            d_A_row_offsets, d_A_col_ids, A_rows, A_nnz,
            grid_h, grid_w, tile_rows, tile_cols, d_density
        );
        CHECK_CUDA_KERNEL();
    }
    cudaEventRecord(e1, stream);

    // ------- Stage 2: mean / stddev + threshold + init labels -------
    // Reduce via cub: sum and sum-of-squares (cast to double on host after).
    uint64_t* d_sum   = nullptr;
    uint64_t* d_sumsq = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_sum,   sizeof(uint64_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_sumsq, sizeof(uint64_t), stream));

    // Sum density tiles -> uint64 (use TransformInputIterator to widen the
    // accumulator type and avoid 32-bit overflow on large matrices).
    {
        cub::TransformInputIterator<uint64_t, DensityCastOp, const uint32_t*> it(d_density, DensityCastOp());
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes, it, d_sum, num_tiles, stream);
        void* d_tmp = nullptr;
        CHECK_CUDA(cudaMallocAsync(&d_tmp, tmp_bytes, stream));
        cub::DeviceReduce::Sum(d_tmp, tmp_bytes, it, d_sum, num_tiles, stream);
        CHECK_CUDA(cudaFreeAsync(d_tmp, stream));
    }

    // Sum-of-squares: cub TransformInputIterator with a functor.
    {
        cub::TransformInputIterator<uint64_t, DensitySqOp, const uint32_t*> it(d_density, DensitySqOp());
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes, it, d_sumsq, num_tiles, stream);
        void* d_tmp = nullptr;
        CHECK_CUDA(cudaMallocAsync(&d_tmp, tmp_bytes, stream));
        cub::DeviceReduce::Sum(d_tmp, tmp_bytes, it, d_sumsq, num_tiles, stream);
        CHECK_CUDA(cudaFreeAsync(d_tmp, stream));
    }

    uint64_t h_sum = 0, h_sumsq = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_sum,   d_sum,   sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&h_sumsq, d_sumsq, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    double mean_d   = (double)h_sum / (double)num_tiles;
    double mean_sq  = (double)h_sumsq / (double)num_tiles;
    double var_d    = mean_sq - mean_d * mean_d;
    if (var_d < 0.0) var_d = 0.0;
    double stddev_d = sqrt(var_d);
    out.density_mean   = mean_d;
    out.density_stddev = stddev_d;

    uint32_t* d_labels = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_labels, sizeof(uint32_t) * num_tiles, stream));
    {
        constexpr int BS = 128;
        size_t nblocks = (num_tiles + BS - 1) / BS;
        thresholdAndInitLabelsKernel<<<(unsigned int)nblocks, BS, 0, stream>>>(
            d_density, d_labels, num_tiles, mean_d, stddev_d, config.segmentation_hot_threshold_k
        );
        CHECK_CUDA_KERNEL();
    }
    cudaEventRecord(e2, stream);

    // ------- Stage 3: iterative connected components -------
    int* d_changed = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_changed, sizeof(int), stream));

    int it = 0;
    {
        dim3 bs(16, 16);
        dim3 nb((grid_w + bs.x - 1) / bs.x, (grid_h + bs.y - 1) / bs.y);
        for (; it < config.segmentation_cc_max_iters; ++it) {
            CHECK_CUDA(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));
            ccPropagationKernel<<<nb, bs, 0, stream>>>(d_labels, grid_h, grid_w, d_changed);
            CHECK_CUDA_KERNEL();
            int h_changed = 0;
            CHECK_CUDA(cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (!h_changed) break;
        }
    }

    // pointer-jumping flatten (a few rounds are sufficient for shallow chains)
    {
        constexpr int BS = 128;
        size_t nblocks = (num_tiles + BS - 1) / BS;
        for (int pj = 0; pj < 8; ++pj) {
            CHECK_CUDA(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));
            ccPointerJumpKernel<<<(unsigned int)nblocks, BS, 0, stream>>>(d_labels, num_tiles, d_changed);
            CHECK_CUDA_KERNEL();
            int h_changed = 0;
            CHECK_CUDA(cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (!h_changed) break;
        }
    }
    out.cc_iterations = it;
    cudaEventRecord(e3, stream);

    // ------- Stage 4: per-grid-row dominant label -------
    uint32_t* d_row_band_label = nullptr;
    uint32_t* d_row_band_hot   = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_row_band_label, sizeof(uint32_t) * grid_h, stream));
    CHECK_CUDA(cudaMallocAsync(&d_row_band_hot,   sizeof(uint32_t) * grid_h, stream));
    {
        size_t smem_bytes = sizeof(uint32_t) * 2 * grid_w;
        // pick block size <= 256 to keep things simple; fewer than grid_w threads is fine.
        int bs = 128;
        if ((uint32_t)bs > grid_w) bs = (int)grid_w;
        if (bs < 32) bs = 32;
        rowBandDominantLabelKernel<<<grid_h, bs, smem_bytes, stream>>>(
            d_labels, grid_h, grid_w, d_row_band_label, d_row_band_hot
        );
        CHECK_CUDA_KERNEL();
    }

    std::vector<uint32_t> h_row_band(grid_h);
    std::vector<uint32_t> h_row_hot(grid_h);
    CHECK_CUDA(cudaMemcpyAsync(h_row_band.data(), d_row_band_label, sizeof(uint32_t) * grid_h, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_row_hot.data(),  d_row_band_hot,   sizeof(uint32_t) * grid_h, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // RLE on host: build segments by run-length encoding the per-grid-row label.
    // Cold rows (label == 0xFFFFFFFF) become "background" segments.
    struct RowBand { uint32_t row_start; uint32_t row_end; uint32_t label; };
    std::vector<RowBand> bands;
    {
        uint32_t cur_label = h_row_band[0];
        uint32_t cur_start = 0;
        for (uint32_t g = 1; g < grid_h; ++g) {
            if (h_row_band[g] != cur_label) {
                RowBand b;
                b.row_start = cur_start * tile_rows;
                b.row_end   = std::min((uint32_t)A_rows, g * tile_rows);
                b.label     = cur_label;
                if (b.row_end > b.row_start) bands.push_back(b);
                cur_label = h_row_band[g];
                cur_start = g;
            }
        }
        RowBand b;
        b.row_start = cur_start * tile_rows;
        b.row_end   = (uint32_t)A_rows;
        b.label     = cur_label;
        if (b.row_end > b.row_start) bands.push_back(b);
    }

    // Merge tiny bands (< min_segment_rows) into the previous band.
    {
        std::vector<RowBand> merged;
        uint32_t min_rows = (uint32_t)std::max(1, config.segmentation_min_segment_rows);
        for (auto& b : bands) {
            if (!merged.empty() && (b.row_end - b.row_start) < min_rows) {
                merged.back().row_end = b.row_end;
            } else {
                merged.push_back(b);
            }
        }
        bands.swap(merged);
    }

    // Count distinct non-cold labels for reporting.
    {
        std::vector<uint32_t> uniq;
        for (auto& b : bands) {
            if (b.label != 0xFFFFFFFFu &&
                std::find(uniq.begin(), uniq.end(), b.label) == uniq.end()) {
                uniq.push_back(b.label);
            }
        }
        out.num_components = (uint32_t)uniq.size();
    }

    // ------- Stage 5: per-segment reduce + classify on GPU -------
    uint32_t num_segments = (uint32_t)bands.size();
    if (num_segments == 0) {
        // entire matrix in one segment as a fallback
        RowBand b{0, (uint32_t)A_rows, 0xFFFFFFFFu};
        bands.push_back(b);
        num_segments = 1;
    }

    std::vector<uint32_t> h_seg_starts(num_segments);
    std::vector<uint32_t> h_seg_ends(num_segments);
    for (uint32_t i = 0; i < num_segments; ++i) {
        h_seg_starts[i] = bands[i].row_start;
        h_seg_ends[i]   = bands[i].row_end;
    }

    uint32_t* d_seg_start = nullptr;
    uint32_t* d_seg_end   = nullptr;
    SegmentRecord* d_records = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_seg_start, sizeof(uint32_t) * num_segments, stream));
    CHECK_CUDA(cudaMallocAsync(&d_seg_end,   sizeof(uint32_t) * num_segments, stream));
    CHECK_CUDA(cudaMallocAsync(&d_records,   sizeof(SegmentRecord) * num_segments, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_seg_start, h_seg_starts.data(), sizeof(uint32_t) * num_segments, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_seg_end,   h_seg_ends.data(),   sizeof(uint32_t) * num_segments, cudaMemcpyHostToDevice, stream));

    // classification thresholds (heuristics; tunable):
    //   dense if avg row produces filling > 25% of its column span
    //   power-law if row-length CV >= 1.5 (heavy tail)
    //   banded if column span < 5% of B_cols and ER >= input_expansion_threshold
    double dense_density_threshold = 0.25;
    double power_law_cv_threshold  = 1.5;
    double banded_col_span_ratio   = 0.05;

    {
        constexpr int BS = 128;
        perSegmentReduceClassifyKernel<<<num_segments, BS, 0, stream>>>(
            d_seg_start, d_seg_end,
            d_num_products, d_max_b_row_len, d_avg_b_row_len,
            d_b_col_idx_left, d_b_col_idx_right,
            d_A_row_offsets, A_rows, A_nnz, (uint32_t)B_cols,
            d_records,
            dense_density_threshold,
            power_law_cv_threshold,
            banded_col_span_ratio,
            config.estimation_input_expansion_threshold
        );
        CHECK_CUDA_KERNEL();
    }

    out.segments.resize(num_segments);
    CHECK_CUDA(cudaMemcpyAsync(out.segments.data(), d_records,
        sizeof(SegmentRecord) * num_segments, cudaMemcpyDeviceToHost, stream));

    cudaEventRecord(e4, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEventElapsedTime(&out.time_density_ms,   e0, e1);
    cudaEventElapsedTime(&out.time_threshold_ms, e1, e2);
    cudaEventElapsedTime(&out.time_cc_ms,        e2, e3);
    cudaEventElapsedTime(&out.time_classify_ms,  e3, e4);

    // cleanup
    CHECK_CUDA(cudaFreeAsync(d_density,        stream));
    CHECK_CUDA(cudaFreeAsync(d_sum,            stream));
    CHECK_CUDA(cudaFreeAsync(d_sumsq,          stream));
    CHECK_CUDA(cudaFreeAsync(d_labels,         stream));
    CHECK_CUDA(cudaFreeAsync(d_changed,        stream));
    CHECK_CUDA(cudaFreeAsync(d_row_band_label, stream));
    CHECK_CUDA(cudaFreeAsync(d_row_band_hot,   stream));
    CHECK_CUDA(cudaFreeAsync(d_seg_start,      stream));
    CHECK_CUDA(cudaFreeAsync(d_seg_end,        stream));
    CHECK_CUDA(cudaFreeAsync(d_records,        stream));

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4);
}

inline const char* segmentClassName(uint32_t cls) {
    switch (cls) {
        case SEG_SPARSE_UNIFORM: return "SPARSE_UNIFORM";
        case SEG_DENSE:          return "DENSE";
        case SEG_POWER_LAW:      return "POWER_LAW";
        case SEG_BANDED:         return "BANDED";
        default:                 return "UNKNOWN";
    }
}

}  // namespace somespgemm
