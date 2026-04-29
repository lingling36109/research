#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "AccumulatorHash.cuh"
#include "Common.h"
#include "CSR.h"

namespace somespgemm {

    void analysisKernelWrapper (
        index_t *A_row_offsets, index_t *A_col_ids, size_t A_rows, size_t A_nnz,
        index_t *B_row_offsets, index_t *B_col_ids, size_t B_rows, size_t B_nnz,
        index_t *num_product,
        index_t *max_b_row_len,
        index_t *avg_b_row_len,
        index_t *b_col_idx_left,
        index_t *b_col_idx_right,
        index_t *max_a_row_nnz,
        cudaStream_t &stream
    ) {
        double avg_nnz_per_row = 1.0 * A_nnz / A_rows;
        if (avg_nnz_per_row < 16) {
            constexpr int CONCURRENT_ROWS = 16;
            constexpr int rows_per_block = CONCURRENT_ROWS * PRODUCT_KERNEL_ITER_PER_BLOCK;
            size_t nblocks = (A_rows + rows_per_block - 1) / rows_per_block;
            analysisKernel<CONCURRENT_ROWS><<<nblocks, PRODUCT_KERNEL_BLOCK_SIZE, 0, stream>>>(
                A_row_offsets, A_col_ids, A_rows, A_nnz,
                B_row_offsets, B_col_ids, B_rows, B_nnz,
                num_product,
                max_b_row_len,
                avg_b_row_len,
                b_col_idx_left,
                b_col_idx_right,
                max_a_row_nnz
            );
        } else {
            constexpr int CONCURRENT_ROWS = 4;
            constexpr int rows_per_block = CONCURRENT_ROWS * PRODUCT_KERNEL_ITER_PER_BLOCK;
            size_t nblocks = (A_rows + rows_per_block - 1) / rows_per_block;
            analysisKernel<CONCURRENT_ROWS><<<nblocks, PRODUCT_KERNEL_BLOCK_SIZE, 0, stream>>>(
                A_row_offsets, A_col_ids, A_rows, A_nnz,
                B_row_offsets, B_col_ids, B_rows, B_nnz,
                num_product,
                max_b_row_len,
                avg_b_row_len,
                b_col_idx_left,
                b_col_idx_right,
                max_a_row_nnz
            );
        }

    }

    template<bool CHECK_OVERFLOW=false>
    void hashSymbolicLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        std::vector<index_t> &h_bin_count,
        index_t* bins,
        index_t* d_num_products,
        index_t* d_max_b_row_len,
        index_t h_max_product,
        index_t* &d_global_mem_buffer,
        index_t* d_out_nnz_per_row,
        size_t sample_compaction_ratio,
        bool always_use_gmem,
        std::vector<cudaStream_t> &streams,
        index_t* overflow_buffer=nullptr,
        uint32_t* overflow_buffer_bitmap=nullptr,
        int num_global_buffers=0,
        size_t global_buffer_size=0
    ) {
        if (h_bin_count[6] != 0) {
            index_t hash_size = h_max_product * HASH_SYMBOLIC_GMEM_SIZE_EXPAND_COE;
            index_t num_blocks = h_bin_count[6];
            if (num_blocks > LARGEST_KERNEL_NUM_BLOCKS) num_blocks = LARGEST_KERNEL_NUM_BLOCKS;

            size_t alloc_size = sizeof(index_t) * num_blocks * hash_size;
            CHECK_CUDA(cudaMallocAsync(&d_global_mem_buffer, alloc_size, streams[6]));
            CHECK_CUDA(cudaMemsetAsync(d_global_mem_buffer, 0xFF, alloc_size, streams[6]));

            cudaFuncSetAttribute(hashSymbolicGMEMKernel<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            hashSymbolicGMEMKernel<10><<<num_blocks,  BLOCK_SIZES[6], SHARED_MEM_PER_BLOCK[6], streams[6]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 6 * A->rows, 
                d_out_nnz_per_row,
                d_global_mem_buffer,
                HASH_BIN_SIZES[6],
                hash_size,
                h_bin_count[6],
                sample_compaction_ratio,
                always_use_gmem
            );
            CHECK_CUDA_KERNEL();
        }

        cudaFuncSetAttribute(hashSymbolicKernel<10, HASH_BIN_SIZES[5], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[5] != 0)
            hashSymbolicKernel<10, HASH_BIN_SIZES[5], CHECK_OVERFLOW><<<h_bin_count[5], BLOCK_SIZES[5], SHARED_MEM_PER_BLOCK[5], streams[5]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 5 * A->rows, 
                d_out_nnz_per_row,
                overflow_buffer,
                overflow_buffer_bitmap,
                num_global_buffers,
                global_buffer_size
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashSymbolicKernel<9, HASH_BIN_SIZES[4], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[4] != 0)
            hashSymbolicKernel<9, HASH_BIN_SIZES[4], CHECK_OVERFLOW><<<h_bin_count[4], BLOCK_SIZES[4], SHARED_MEM_PER_BLOCK[4], streams[4]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 4 * A->rows, 
                d_out_nnz_per_row,
                overflow_buffer,
                overflow_buffer_bitmap,
                num_global_buffers,
                global_buffer_size
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashSymbolicKernel<8, HASH_BIN_SIZES[3], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[3] != 0)
            hashSymbolicKernel<8, HASH_BIN_SIZES[3], CHECK_OVERFLOW><<<h_bin_count[3], BLOCK_SIZES[3], SHARED_MEM_PER_BLOCK[3], streams[3]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 3 * A->rows, 
                d_out_nnz_per_row,
                overflow_buffer,
                overflow_buffer_bitmap,
                num_global_buffers,
                global_buffer_size
            );

        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashSymbolicKernel<7, HASH_BIN_SIZES[2], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[2] != 0)
            hashSymbolicKernel<7, HASH_BIN_SIZES[2], CHECK_OVERFLOW><<<h_bin_count[2], BLOCK_SIZES[2], SHARED_MEM_PER_BLOCK[2], streams[2]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 2 * A->rows, 
                d_out_nnz_per_row,
                overflow_buffer,
                overflow_buffer_bitmap,
                num_global_buffers,
                global_buffer_size
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashSymbolicKernel<6, HASH_BIN_SIZES[1], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[1] != 0)
            hashSymbolicKernel<6, HASH_BIN_SIZES[1], CHECK_OVERFLOW><<<h_bin_count[1], BLOCK_SIZES[1], SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 1 * A->rows, 
                d_out_nnz_per_row,
                overflow_buffer,
                overflow_buffer_bitmap,
                num_global_buffers,
                global_buffer_size
            );

        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashSymbolicSubWarpKernel<6, HASH_BIN_SIZES[0], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[0] != 0)
            hashSymbolicSubWarpKernel<6, HASH_BIN_SIZES[0], CHECK_OVERFLOW><<<h_bin_count[0] / 2 + 1, BLOCK_SIZES[0], SHARED_MEM_PER_BLOCK[0], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 0 * A->rows,
                d_out_nnz_per_row,
                h_bin_count[0],
                overflow_buffer,
                overflow_buffer_bitmap,
                num_global_buffers,
                global_buffer_size
            );

        CHECK_CUDA_KERNEL();
    };

    template<bool QUERY_BITMAP=true>
    void denseSymbolicLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        index_t* h_bin_count,
        index_t* bins,
        index_t* d_num_products,
        index_t* d_max_b_row_len,
        index_t* d_b_col_idx_left,
        index_t* d_b_col_idx_right,
        index_t h_max_product,
        index_t* d_out_nnz_per_row,
        size_t sample_compaction_ratio,
        cudaStream_t* streams
    ) {
        if (h_bin_count[6] != 0 || h_bin_count[0] !=0 ) {
            throw std::runtime_error("Smallest and largest bins not supported!");
        }

        // launch the dense kernels
        cudaFuncSetAttribute(denseSymbolicKernel<10, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[5] != 0)
            denseSymbolicKernel<10, QUERY_BITMAP><<<h_bin_count[5], BLOCK_SIZES[5], SHARED_MEM_PER_BLOCK[5], streams[5]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_b_col_idx_left, d_b_col_idx_right,
                bins + 5 * A->rows,
                d_out_nnz_per_row,
                DENSE_BIN_SIZES[5]
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseSymbolicKernel<9, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[4] != 0)
            denseSymbolicKernel<9, QUERY_BITMAP><<<h_bin_count[4], BLOCK_SIZES[4], SHARED_MEM_PER_BLOCK[4], streams[4]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_b_col_idx_left, d_b_col_idx_right,
                bins + 4 * A->rows,
                d_out_nnz_per_row,
                DENSE_BIN_SIZES[4]
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseSymbolicKernel<8, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[3] != 0)
            denseSymbolicKernel<8, QUERY_BITMAP><<<h_bin_count[3], BLOCK_SIZES[3], SHARED_MEM_PER_BLOCK[3], streams[3]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_b_col_idx_left, d_b_col_idx_right,
                bins + 3 * A->rows,
                d_out_nnz_per_row,
                DENSE_BIN_SIZES[3]
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseSymbolicKernel<7, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[2] != 0)
            denseSymbolicKernel<7, QUERY_BITMAP><<<h_bin_count[2], BLOCK_SIZES[2], SHARED_MEM_PER_BLOCK[2], streams[2]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_b_col_idx_left, d_b_col_idx_right,
                bins + 2 * A->rows,
                d_out_nnz_per_row,
                DENSE_BIN_SIZES[2]
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseSymbolicKernel<6, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[1] != 0)
            denseSymbolicKernel<6, QUERY_BITMAP><<<h_bin_count[1], BLOCK_SIZES[1], SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_b_col_idx_left, d_b_col_idx_right,
                bins + 1 * A->rows,
                d_out_nnz_per_row,
                DENSE_BIN_SIZES[1]
            );
        CHECK_CUDA_KERNEL();
    }

    void sparseOutlierLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        int spark_outlier_kernel_size,
        int spark_kernel_size,
        size_t* d_num_products_prefixsum,
        index_t* d_num_products,
        index_t* d_max_b_row_len,
        index_t* d_b_col_idx_left,
        index_t* d_b_col_idx_right,
        index_t* d_outlier_rows,
        index_t h_num_outlier_rows,
        index_t* d_out_estik_indices,
        data_t* d_out_estik_values,
        index_t* d_out_num_outputs,
        index_t* &d_global_mem_buffer,
        index_t* &d_global_mem_bitmap,
        cudaStream_t* streams
    ) {
        if (spark_outlier_kernel_size <= 128) {
            size_t nthreads = 64;
            index_t nblocks = (h_num_outlier_rows + 1) / 2;
            hashNumericSubWarpKernel<5, HASH_NUMERIC_USPARSE_UPPER_BOUNDS[1]><<<nblocks, nthreads, SHARED_MEM_PER_BLOCK[1], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products,
                d_outlier_rows,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs,
                (index_t)128
            );
        } else if (spark_outlier_kernel_size <= HASH_NUMERIC_BIN_SIZES[1] * 0.85) {
            cudaFuncSetAttribute(hashNumericKernel<6, HASH_NUMERIC_BIN_SIZES[1], false, false, size_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            hashNumericKernel<6, HASH_NUMERIC_BIN_SIZES[1], false, false, size_t><<<h_num_outlier_rows, BLOCK_SIZES[1], SHARED_MEM_PER_BLOCK[1], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_outlier_rows,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs
            );
        } else if (spark_outlier_kernel_size <= HASH_NUMERIC_BIN_SIZES[2] * 0.85) {
            cudaFuncSetAttribute(hashNumericKernel<7, HASH_NUMERIC_BIN_SIZES[2], false, false, size_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            hashNumericKernel<7, HASH_NUMERIC_BIN_SIZES[2], false, false, size_t><<<h_num_outlier_rows, BLOCK_SIZES[2], SHARED_MEM_PER_BLOCK[2], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_outlier_rows,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs
            );
        } else if (spark_outlier_kernel_size <= HASH_NUMERIC_BIN_SIZES[3] * 0.85) {
            cudaFuncSetAttribute(hashNumericKernel<8, HASH_NUMERIC_BIN_SIZES[3], false, false, size_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            hashNumericKernel<8, HASH_NUMERIC_BIN_SIZES[3], false, false, size_t><<<h_num_outlier_rows, BLOCK_SIZES[3], SHARED_MEM_PER_BLOCK[3], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_outlier_rows,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs
            );
        } else if (spark_outlier_kernel_size <= HASH_NUMERIC_BIN_SIZES[4] * 0.85) {
            cudaFuncSetAttribute(hashNumericKernel<9, HASH_NUMERIC_BIN_SIZES[4], false, false, size_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            hashNumericKernel<9, HASH_NUMERIC_BIN_SIZES[4], false, false, size_t><<<h_num_outlier_rows, BLOCK_SIZES[4], SHARED_MEM_PER_BLOCK[4], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_outlier_rows,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs
            );
        } else if (spark_outlier_kernel_size <= HASH_NUMERIC_BIN_SIZES[5] * 0.85) {
            cudaFuncSetAttribute(hashNumericKernel<10, HASH_NUMERIC_BIN_SIZES[5], false, false, size_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            hashNumericKernel<10, HASH_NUMERIC_BIN_SIZES[5], false, false, size_t><<<h_num_outlier_rows, BLOCK_SIZES[5], SHARED_MEM_PER_BLOCK[5], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                d_outlier_rows,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs
            );
        } else {
            index_t num_buffers = LARGEST_KERNEL_NUM_BUFFERS;
            size_t shared_mem_size = sizeof(index_t) * num_buffers * B->rows;

            CHECK_CUDA(cudaMallocAsync(&d_global_mem_bitmap, sizeof(uint32_t) * num_buffers, streams[0]));
            CHECK_CUDA(cudaMallocAsync(&d_global_mem_buffer, shared_mem_size, streams[0]));
            CHECK_CUDA(cudaMemsetAsync(d_global_mem_bitmap, 0x00, sizeof(uint32_t) * num_buffers, streams[0]));
            
            index_t* global_mem_buffer = d_global_mem_buffer;
            cudaFuncSetAttribute(denseNumericIterKernel<10, false, true, size_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericIterKernel<10, false, true, size_t><<<h_num_outlier_rows, DENSE_NUMERIC_LARGEST_BLOCK_SIZE, DENSE_NUMERIC_LARGEST_SMEM_SIZE, streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values,
                d_num_products,
                d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                d_outlier_rows,
                DENSE_NUMERIC_LARGEST_ITER_SIZE,
                // h_num_outlier_rows,
                DENSE_NUMERIC_LARGEST_ROW_OFFSET_SMEM,
                global_mem_buffer,
                d_global_mem_bitmap,
                d_out_num_outputs
            );

        }
        CHECK_CUDA_KERNEL();
    }

    void sparseKernelLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        int spark_kernel_size,
        size_t* d_num_products_prefixsum,
        index_t* d_num_products,
        index_t* d_out_estik_indices,
        data_t* d_out_estik_values,
        index_t* d_out_num_outputs,
        index_t* d_output_col_range_l,
        index_t* d_output_col_range_r,
        bool use_esc,
        std::vector<cudaStream_t> &streams
    ) {

        CHECK_CUDA_KERNEL();

        if (spark_kernel_size == 16) {
            if (use_esc) {
                size_t nthreads = spark_kernel_size <= NUMERIC_USPARSE_K_BLOCK_SIZE ? NUMERIC_USPARSE_K_BLOCK_SIZE : spark_kernel_size;
                size_t nblocks = (A->rows + NUMERIC_USPARSE_K_ROWS_PER_BLOCK- 1) / NUMERIC_USPARSE_K_ROWS_PER_BLOCK;
                ESCKernelDispatcher<16><<<nblocks, NUMERIC_USPARSE_K_BLOCK_SIZE, 0, streams[1]>>>(
                    A->row_offsets, A->col_ids, A->data, A->rows,
                    B->row_offsets, B->col_ids, B->data, B->rows,
                    d_out_estik_indices, d_out_estik_values, d_out_num_outputs,
                    d_num_products_prefixsum, d_num_products, NUMERIC_USPARSE_K_ROWS_PER_BLOCK
                );
            } else {
                size_t nthreads = 64;
                size_t nblocks = (A->rows + 3) / 4;
                hashNumericSubWarpKernel<4, HASH_NUMERIC_USPARSE_UPPER_BOUNDS[0]><<<nblocks, nthreads, SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                    A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                    B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                    d_num_products,
                    nullptr,
                    d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                    d_out_num_outputs,
                    (index_t)16
                );
            }
        }
        else if (spark_kernel_size == 32) {
            if (use_esc) {
                size_t nthreads = spark_kernel_size <= NUMERIC_USPARSE_K_BLOCK_SIZE ? NUMERIC_USPARSE_K_BLOCK_SIZE : spark_kernel_size;
                size_t nblocks = (A->rows + NUMERIC_USPARSE_K_ROWS_PER_BLOCK - 1) / NUMERIC_USPARSE_K_ROWS_PER_BLOCK;
                ESCKernelDispatcher<32><<<nblocks, NUMERIC_USPARSE_K_BLOCK_SIZE, 0, streams[1]>>>(
                    A->row_offsets, A->col_ids, A->data, A->rows,
                    B->row_offsets, B->col_ids, B->data, B->rows,
                    d_out_estik_indices, d_out_estik_values, d_out_num_outputs,
                    d_num_products_prefixsum, d_num_products, NUMERIC_USPARSE_K_ROWS_PER_BLOCK
                );
            } else {
                size_t nthreads = 64;
                size_t nblocks = (A->rows + 3) / 4;
                hashNumericSubWarpKernel<4, HASH_NUMERIC_USPARSE_UPPER_BOUNDS[0]><<<nblocks, nthreads, SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                    A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                    B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                    d_num_products,
                    nullptr,
                    d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                    d_out_num_outputs,
                    (index_t)32
                );
            }
        }
        else if (spark_kernel_size == 64) {
            size_t nthreads = 64;
            size_t nblocks = (A->rows + 3) / 4;
            hashNumericSubWarpKernel<4, HASH_NUMERIC_USPARSE_UPPER_BOUNDS[0]><<<nblocks, nthreads, SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products,
                nullptr,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs,
                (index_t)64
            );
        }
        else if (spark_kernel_size == 128) {
            size_t nthreads = 64;
            size_t nblocks = (A->rows + 1) / 2;
            hashNumericSubWarpKernel<5, HASH_NUMERIC_USPARSE_UPPER_BOUNDS[1]><<<nblocks, nthreads, SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products,
                nullptr,
                d_num_products_prefixsum, d_out_estik_indices, d_out_estik_values, 
                d_out_num_outputs,
                (index_t)128
            );
        }
        else
            throw std::runtime_error("Unsupported ultra-sparse kernel size");

        CHECK_CUDA_KERNEL();
    }

    template<bool CHECK_OVERFLOW>
    void hashNumericLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        std::vector<index_t> &h_bin_count,
        index_t* bins,
        index_t* d_num_products,
        index_t* d_max_b_row_len,
        index_t* d_output_row_offsets,
        index_t* d_out_indices, data_t* d_out_values, 
        std::vector<cudaStream_t> &streams,
        index_t *d_out_nnz = nullptr,
        index_t* d_num_overflow_rows = nullptr,
        index_t* d_overflow_rows = nullptr,
        index_t* d_overflow_buffer_allocated=nullptr,
        data_t* d_global_hashmap_buffer=nullptr,
        index_t* d_global_buffer_bitmap=nullptr,
        index_t num_global_buffers=0
    ) {

        if (h_bin_count[6] != 0) {
            // std::cout << "Largest bin for hash Numeric Kernels have bin count " << h_bin_count[6] << std::endl;
            // std::cout << "This is skipped currently as not supported!" << std::endl;
            cudaFuncSetAttribute(hashNumericKernel<10, HASH_NUMERIC_BIN_SIZES[6], CHECK_OVERFLOW, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            if (h_bin_count[6] != 0)
                hashNumericKernel<10, HASH_NUMERIC_BIN_SIZES[6], CHECK_OVERFLOW, true><<<h_bin_count[6], BLOCK_SIZES[6], SHARED_MEM_PER_BLOCK[6], streams[6]>>>(
                    A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                    B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                    d_num_products, d_max_b_row_len,
                    bins + 6 * A->rows,
                    d_output_row_offsets,
                    d_out_indices, d_out_values,
                    d_out_nnz,
                    d_num_overflow_rows,
                    d_overflow_rows,
                    d_overflow_buffer_allocated,
                    d_global_hashmap_buffer,
                    d_global_buffer_bitmap,
                    num_global_buffers,
                    HASH_NUMERIC_BIN_SIZES[6]
                );
            CHECK_CUDA_KERNEL();
        }

        cudaFuncSetAttribute(hashNumericKernel<10, HASH_NUMERIC_BIN_SIZES[5], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[5] != 0)
            hashNumericKernel<10, HASH_NUMERIC_BIN_SIZES[5], CHECK_OVERFLOW><<<h_bin_count[5], BLOCK_SIZES[5], SHARED_MEM_PER_BLOCK[5], streams[5]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 5 * A->rows,
                d_output_row_offsets,
                d_out_indices, d_out_values,
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashNumericKernel<9, HASH_NUMERIC_BIN_SIZES[4], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[4] != 0)
            hashNumericKernel<9, HASH_NUMERIC_BIN_SIZES[4], CHECK_OVERFLOW><<<h_bin_count[4], BLOCK_SIZES[4], SHARED_MEM_PER_BLOCK[4], streams[4]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 4 * A->rows,
                d_output_row_offsets,
                d_out_indices, d_out_values,
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashNumericKernel<8, HASH_NUMERIC_BIN_SIZES[3], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[3] != 0)
            hashNumericKernel<8, HASH_NUMERIC_BIN_SIZES[3], CHECK_OVERFLOW><<<h_bin_count[3], BLOCK_SIZES[3], SHARED_MEM_PER_BLOCK[3], streams[3]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 3 * A->rows,
                d_output_row_offsets,
                d_out_indices, d_out_values,
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashNumericKernel<7, HASH_NUMERIC_BIN_SIZES[2], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[2] != 0)
            hashNumericKernel<7, HASH_NUMERIC_BIN_SIZES[2], CHECK_OVERFLOW><<<h_bin_count[2], BLOCK_SIZES[2], SHARED_MEM_PER_BLOCK[2], streams[2]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 2 * A->rows,
                d_output_row_offsets,
                d_out_indices, d_out_values,
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(hashNumericKernel<6, HASH_NUMERIC_BIN_SIZES[1], CHECK_OVERFLOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[1] != 0)
            hashNumericKernel<6, HASH_NUMERIC_BIN_SIZES[1], CHECK_OVERFLOW><<<h_bin_count[1], BLOCK_SIZES[1], SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_num_products, d_max_b_row_len,
                bins + 1 * A->rows,
                d_output_row_offsets,
                d_out_indices, d_out_values,
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        if (h_bin_count[0] != 0) {
            std::cout << "Not supported smallest bin for hash Numeric Kernels! The bin count is " << h_bin_count[0] << std::endl;
            throw std::runtime_error("Not supported smallest bin for hash kernel!");
        }
    }


    template<bool CHECK_OVERFLOW, bool QUERY_BITMAP=false>
    void denseNumericLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        index_t* h_bin_count,
        index_t* bins,
        index_t* d_num_products,
        index_t* d_max_b_row_len,
        index_t* d_b_col_idx_left,
        index_t* d_b_col_idx_right,
        index_t* d_output_row_offsets,
        index_t* d_out_indices, data_t* d_out_values, 
        cudaStream_t *streams,
        index_t* global_mem_buffer,
        index_t buffer_size,
        index_t *global_bitmap,
        index_t *d_out_nnz = nullptr,
        index_t *d_max_nnz = nullptr,
        index_t* d_num_overflow_rows = nullptr,
        index_t* d_overflow_rows = nullptr,
        index_t* d_overflow_buffer_allocated = nullptr
    ) {
        cudaFuncSetAttribute(denseNumericKernel<10, CHECK_OVERFLOW, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[5] != 0)
            denseNumericKernel<10, CHECK_OVERFLOW, QUERY_BITMAP><<<h_bin_count[5], BLOCK_SIZES[5], SHARED_MEM_PER_BLOCK[5], streams[5]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 5 * A->rows,
                DENSE_NUMERIC_BIN_SIZES[5],
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseNumericKernel<9, CHECK_OVERFLOW, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[4] != 0)
            denseNumericKernel<9, CHECK_OVERFLOW, QUERY_BITMAP><<<h_bin_count[4], BLOCK_SIZES[4], SHARED_MEM_PER_BLOCK[4], streams[4]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 4 * A->rows,
                DENSE_NUMERIC_BIN_SIZES[4],
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();


        cudaFuncSetAttribute(denseNumericKernel<8, CHECK_OVERFLOW, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[3] != 0)
            denseNumericKernel<8, CHECK_OVERFLOW, QUERY_BITMAP><<<h_bin_count[3], BLOCK_SIZES[3], SHARED_MEM_PER_BLOCK[3], streams[3]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 3 * A->rows,
                DENSE_NUMERIC_BIN_SIZES[3],
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();


        cudaFuncSetAttribute(denseNumericKernel<7, CHECK_OVERFLOW, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[2] != 0)
            denseNumericKernel<7, CHECK_OVERFLOW, QUERY_BITMAP><<<h_bin_count[2], BLOCK_SIZES[2], SHARED_MEM_PER_BLOCK[2], streams[2]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 2 * A->rows,
                DENSE_NUMERIC_BIN_SIZES[2],
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseNumericKernel<6, CHECK_OVERFLOW, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[1] != 0)
            denseNumericKernel<6, CHECK_OVERFLOW, QUERY_BITMAP><<<h_bin_count[1], BLOCK_SIZES[1], SHARED_MEM_PER_BLOCK[1], streams[1]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 1 * A->rows,
                DENSE_NUMERIC_BIN_SIZES[1],
                d_out_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );
        CHECK_CUDA_KERNEL();

        cudaFuncSetAttribute(denseNumericSubWarpDispatcher<DENSE_NUMERIC_SMALLEST_LOG_NTHR, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
        if (h_bin_count[0] != 0) {
            int num_blocks = (h_bin_count[0] + 1) / 2;
            denseNumericSubWarpDispatcher<DENSE_NUMERIC_SMALLEST_LOG_NTHR, QUERY_BITMAP><<<num_blocks, BLOCK_SIZES[0], SHARED_MEM_PER_BLOCK[0], streams[0]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 0 * A->rows,
                h_bin_count[0],
                DENSE_NUMERIC_BIN_SIZES[0],
                d_out_nnz
            );
        }
        CHECK_CUDA_KERNEL();

        if (h_bin_count[6] != 0) {
            index_t num_buffers = LARGEST_KERNEL_NUM_BUFFERS;
            index_t required_buffer_size = B->rows * num_buffers * sizeof(index_t);        // this can be smaller
            if (buffer_size < required_buffer_size) {
                std::cout << "Insufficient global memory buffer for largest dense numeric kernel! Required size: " << required_buffer_size << ", provided size: " << buffer_size << std::endl;
                throw std::runtime_error("Insufficient global memory buffer for largest dense numeric kernel!");
            }

            cudaFuncSetAttribute(denseNumericIterKernel<10, CHECK_OVERFLOW, QUERY_BITMAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericIterKernel<10, CHECK_OVERFLOW, QUERY_BITMAP><<<h_bin_count[6], DENSE_NUMERIC_LARGEST_BLOCK_SIZE, DENSE_NUMERIC_LARGEST_SMEM_SIZE, streams[6]>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                d_output_row_offsets, d_out_indices, d_out_values,
                d_num_products, d_max_b_row_len, d_b_col_idx_left, d_b_col_idx_right,
                bins + 6 * A->rows,
                DENSE_NUMERIC_LARGEST_ITER_SIZE,
                // h_bin_count[6],
                DENSE_NUMERIC_LARGEST_ROW_OFFSET_SMEM,
                global_mem_buffer,
                global_bitmap,
                d_out_nnz,
                d_max_nnz,
                d_num_overflow_rows,
                d_overflow_rows,
                d_overflow_buffer_allocated
            );

            CHECK_CUDA_KERNEL();
        }

    };


    void outputSortingLauncher(
        index_t* d_src_indices, data_t* d_src_values, index_t* d_src_row_offsets,
        index_t* d_dst_indices, data_t* d_dst_values, index_t* d_dst_row_offsets,
        index_t* h_num_rows,
        index_t* d_bins,
        index_t* d_num_elements,
        std::vector<cudaStream_t> &streams,
        index_t num_rows_a,
        int sort_min_bin = 3
    ) {
        throw std::runtime_error("Disabled outputSortingLauncher, use outputSortingDynLauncher instead!");
    }

    void denseNumericStaticLauncher(
        std::shared_ptr<cuCSR> A,
        std::shared_ptr<cuCSR> B,
        index_t* d_out_indices, data_t* d_out_values,
        index_t log_nthr_per_a_elem,
        index_t kernel_id,
        cudaStream_t stream,
        index_t* d_out_nnz_per_row = nullptr
    ) {
        // Determine accumulator size based on kernel_id
        index_t accumulator_size;
        if (kernel_id >= 1 && kernel_id <= 5) {
            accumulator_size = DENSE_NUMERIC_BIN_SIZES[kernel_id];
        } else {
            throw std::runtime_error("Invalid kernel_id for denseNumericStaticKernel: " + std::to_string(kernel_id));
        }

        size_t shared_mem_size = SHARED_MEM_PER_BLOCK[kernel_id];
        index_t output_row_size = B->cols;
        index_t num_rows = A->rows;

        if (kernel_id == 1) {
            cudaFuncSetAttribute(denseNumericStaticKernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericStaticKernel<6><<<num_rows, BLOCK_SIZES[kernel_id], shared_mem_size, stream>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                output_row_size, d_out_indices, d_out_values,
                accumulator_size, log_nthr_per_a_elem, d_out_nnz_per_row
            );
        } else if (kernel_id == 2) {
            cudaFuncSetAttribute(denseNumericStaticKernel<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericStaticKernel<7><<<num_rows, BLOCK_SIZES[kernel_id], shared_mem_size, stream>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                output_row_size, d_out_indices, d_out_values,
                accumulator_size, log_nthr_per_a_elem, d_out_nnz_per_row
            );
        } else if (kernel_id == 3) {
            cudaFuncSetAttribute(denseNumericStaticKernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericStaticKernel<8><<<num_rows, BLOCK_SIZES[kernel_id], shared_mem_size, stream>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                output_row_size, d_out_indices, d_out_values,
                accumulator_size, log_nthr_per_a_elem, d_out_nnz_per_row
            );
        } else if (kernel_id == 4) {
            cudaFuncSetAttribute(denseNumericStaticKernel<9>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericStaticKernel<9><<<num_rows, BLOCK_SIZES[kernel_id], shared_mem_size, stream>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                output_row_size, d_out_indices, d_out_values,
                accumulator_size, log_nthr_per_a_elem, d_out_nnz_per_row
            );
        } else if (kernel_id == 5) {
            cudaFuncSetAttribute(denseNumericStaticKernel<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            denseNumericStaticKernel<10><<<num_rows, BLOCK_SIZES[kernel_id], shared_mem_size, stream>>>(
                A->row_offsets, A->col_ids, A->data, A->rows, A->cols, A->nnz,
                B->row_offsets, B->col_ids, B->data, B->rows, B->cols, B->nnz,
                output_row_size, d_out_indices, d_out_values,
                accumulator_size, log_nthr_per_a_elem, d_out_nnz_per_row
            );
        } else {
            throw std::runtime_error("Unsupported LOG_NTHR for denseNumericStaticKernel: " + std::to_string(kernel_id));
        }
        CHECK_CUDA_KERNEL();
    }

    template<int items_per_thread, int threads_per_block, int bin_id, bool FORCE_FAST>
    void sortingKernelWrapper(
        index_t num_rows,
        cudaStream_t& stream,
        index_t* col_in, data_t* val_in, index_t* offset_in,
        index_t* col_out, data_t* val_out, index_t* offset_out,
        index_t* my_rows, index_t* num_elements,
        index_t col_bits,
        bool disable_fast
    ) {
        if (HASH_NUMERIC_IDX_BITS[bin_id] >= col_bits && !disable_fast) {
            auto block_reduce_temp_bytes = sizeof(
                typename cub::BlockRadixSort<index_t, threads_per_block, items_per_thread>::TempStorage
            );
            auto value_temp_bytes = sizeof(data_t) * threads_per_block * items_per_thread;
            auto smem_size = block_reduce_temp_bytes;
            if (smem_size < value_temp_bytes) {
                smem_size = value_temp_bytes;
            }

            cudaFuncSetAttribute(sortOutputFused<items_per_thread, threads_per_block, HASH_NUMERIC_IDX_BITS[bin_id]>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            sortOutputFused<items_per_thread, threads_per_block, HASH_NUMERIC_IDX_BITS[bin_id]><<<num_rows, threads_per_block, smem_size, stream>>>(
                col_in, val_in, offset_in,
                col_out, val_out, offset_out,
                my_rows, num_elements, col_bits
            );

        } else {

            if (FORCE_FAST) {
                std::cerr << "Error: The bin is too large to use slower sorting" << std::endl;
                throw std::runtime_error("Forced fast sort");
            }

            auto block_reduce_temp_bytes = sizeof(
                typename cub::BlockRadixSort<index_t, threads_per_block, items_per_thread, data_t>::TempStorage
            );
            auto smem_size = std::max(
                block_reduce_temp_bytes,
                (sizeof(index_t) + sizeof(data_t)) * threads_per_block * items_per_thread
            );

            cudaFuncSetAttribute(sortOutputDyn<items_per_thread, threads_per_block>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);
            sortOutputDyn<items_per_thread, threads_per_block><<<num_rows, threads_per_block, smem_size, stream>>>(
                col_in, val_in, offset_in,
                col_out, val_out, offset_out,
                my_rows, num_elements
            );
        }
        CHECK_CUDA_KERNEL();
    }

    // CUB uses dynamically allocated shared memory here.
    void outputSortingDynLauncher(
        index_t* d_src_indices, data_t* d_src_values, index_t* d_src_row_offsets,
        index_t* d_dst_indices, data_t* d_dst_values, index_t* d_dst_row_offsets,
        index_t* h_num_rows,
        index_t* d_bins,
        index_t* d_num_elements,
        std::vector<cudaStream_t> &streams,
        index_t num_rows_a,
        index_t num_cols_b,
        bool disable_fast,
        int sort_min_bin = 3
    ) {
        if (sort_min_bin < 1 || sort_min_bin > 6) {
            std::cout << "Invalid sort_min_bin value: " << sort_min_bin << std::endl;
            throw std::runtime_error("Invalid sort_min_bin value!");
        }

        // log 2 num cols b
        index_t col_bits = bits_needed(num_cols_b);

        if (h_num_rows[6] != 0) {
            constexpr int ID = 6;

            constexpr int threads_per_block = SORT_BLOCK_SIZES[ID];
            constexpr int items_per_thread = SORT_ITEMS_PER_THREAD[ID];
            if (threads_per_block * items_per_thread < HASH_NUMERIC_BIN_SIZES[ID]) {
                std::cout << "Insufficient threads per block and items per thread for sort! threads_per_block: " << threads_per_block << ", items_per_thread: " << items_per_thread << ", max possible bin size: " << HASH_NUMERIC_BIN_SIZES[ID] << std::endl;
                throw std::runtime_error("Insufficient threads per block and items per thread for dynamic shared memory sort!");
            }

            sortingKernelWrapper<items_per_thread, threads_per_block, ID, true>(
                h_num_rows[ID],
                streams[ID],
                d_src_indices, d_src_values, d_src_row_offsets,
                d_dst_indices, d_dst_values, d_dst_row_offsets,
                d_bins + ID * num_rows_a,
                d_num_elements,
                col_bits,
                disable_fast
            );
        }
        if (sort_min_bin > 6) return;

        if (h_num_rows[5] != 0) {
            constexpr int ID = 5;

            constexpr int threads_per_block = SORT_BLOCK_SIZES[ID];
            constexpr int items_per_thread = SORT_ITEMS_PER_THREAD[ID];
            if (threads_per_block * items_per_thread < HASH_NUMERIC_BIN_SIZES[ID]) {
                std::cout << "Insufficient threads per block and items per thread for sort! threads_per_block: " << threads_per_block << ", items_per_thread: " << items_per_thread << ", max possible bin size: " << HASH_NUMERIC_BIN_SIZES[ID] << std::endl;
                throw std::runtime_error("Insufficient threads per block and items per thread for dynamic shared memory sort!");
            }

            sortingKernelWrapper<items_per_thread, threads_per_block, ID, false>(
                h_num_rows[ID],
                streams[ID],
                d_src_indices, d_src_values, d_src_row_offsets,
                d_dst_indices, d_dst_values, d_dst_row_offsets,
                d_bins + ID * num_rows_a,
                d_num_elements,
                col_bits,
                disable_fast
            );
        }

        if (sort_min_bin > 4) return;

        if (h_num_rows[4] != 0) {
            constexpr int ID = 4;

            constexpr int threads_per_block = SORT_BLOCK_SIZES[ID];
            constexpr int items_per_thread = SORT_ITEMS_PER_THREAD[ID];
            if (threads_per_block * items_per_thread < HASH_NUMERIC_BIN_SIZES[ID]) {
                std::cout << "Insufficient threads per block and items per thread for sort! threads_per_block: " << threads_per_block << ", items_per_thread: " << items_per_thread << ", max possible bin size: " << HASH_NUMERIC_BIN_SIZES[ID] << std::endl;
                throw std::runtime_error("Insufficient threads per block and items per thread for dynamic shared memory sort!");
            }

            sortingKernelWrapper<items_per_thread, threads_per_block, ID, false>(
                h_num_rows[ID],
                streams[ID],
                d_src_indices, d_src_values, d_src_row_offsets,
                d_dst_indices, d_dst_values, d_dst_row_offsets,
                d_bins + ID * num_rows_a,
                d_num_elements,
                col_bits,
                disable_fast
            );
        }

        if (sort_min_bin > 3) return;

        if (h_num_rows[3] != 0) {
            constexpr int ID = 3;

            constexpr int threads_per_block = SORT_BLOCK_SIZES[ID];
            constexpr int items_per_thread = SORT_ITEMS_PER_THREAD[ID];
            if (threads_per_block * items_per_thread < HASH_NUMERIC_BIN_SIZES[ID]) {
                std::cout << "Insufficient threads per block and items per thread for sort! threads_per_block: " << threads_per_block << ", items_per_thread: " << items_per_thread << ", max possible bin size: " << HASH_NUMERIC_BIN_SIZES[ID] << std::endl;
                throw std::runtime_error("Insufficient threads per block and items per thread for dynamic shared memory sort!");
            }

            sortingKernelWrapper<items_per_thread, threads_per_block, ID, false>(
                h_num_rows[ID],
                streams[ID],
                d_src_indices, d_src_values, d_src_row_offsets,
                d_dst_indices, d_dst_values, d_dst_row_offsets,
                d_bins + ID * num_rows_a,
                d_num_elements,
                col_bits,
                disable_fast
            );
        }

        if (sort_min_bin > 2) return;

        if (h_num_rows[2] != 0) {
            constexpr int ID = 2;

            constexpr int threads_per_block = SORT_BLOCK_SIZES[ID];
            constexpr int items_per_thread = SORT_ITEMS_PER_THREAD[ID];
            if (threads_per_block * items_per_thread < HASH_NUMERIC_BIN_SIZES[ID]) {
                std::cout << "Insufficient threads per block and items per thread for sort! threads_per_block: " << threads_per_block << ", items_per_thread: " << items_per_thread << ", max possible bin size: " << HASH_NUMERIC_BIN_SIZES[ID] << std::endl;
                throw std::runtime_error("Insufficient threads per block and items per thread for dynamic shared memory sort!");
            }

            sortingKernelWrapper<items_per_thread, threads_per_block, ID, false>(
                h_num_rows[ID],
                streams[ID],
                d_src_indices, d_src_values, d_src_row_offsets,
                d_dst_indices, d_dst_values, d_dst_row_offsets,
                d_bins + ID * num_rows_a,
                d_num_elements,
                col_bits,
                disable_fast
            );
        }
        

        if (sort_min_bin > 1) return;
        
        if (h_num_rows[1] != 0) {
            constexpr int ID = 1;

            constexpr int threads_per_block = SORT_BLOCK_SIZES[ID];
            constexpr int items_per_thread = SORT_ITEMS_PER_THREAD[ID];
            if (threads_per_block * items_per_thread < HASH_NUMERIC_BIN_SIZES[ID]) {
                std::cout << "Insufficient threads per block and items per thread for sort! threads_per_block: " << threads_per_block << ", items_per_thread: " << items_per_thread << ", max possible bin size: " << HASH_NUMERIC_BIN_SIZES[ID] << std::endl;
                throw std::runtime_error("Insufficient threads per block and items per thread for dynamic shared memory sort!");
            }

            sortingKernelWrapper<items_per_thread, threads_per_block, ID, false>(
                h_num_rows[ID],
                streams[ID],
                d_src_indices, d_src_values, d_src_row_offsets,
                d_dst_indices, d_dst_values, d_dst_row_offsets,
                d_bins + ID * num_rows_a,
                d_num_elements,
                col_bits,
                disable_fast
            );
        }
            
    }

    template<typename T, class src_offset_t=index_t>
    void copyKernelLauncher(
        T* dst, T* src, 
        index_t* offset_dst, src_offset_t* offsets_src, 
        index_t* nums, 
        index_t num_rows,
        index_t* copy_rows=nullptr,
        index_t avg_elements_per_row=32,
        cudaStream_t stream=0,
        index_t src_num_cols=0
    ) {
        if (num_rows <= NUM_SM * 8) {
            // in this case we would have iter=1
            constexpr int ITER_PER_BLOCK = 1;

            if (avg_elements_per_row < 16) {
                // in this case we would have threads_per_row = 16
                constexpr int THREADS_PER_ROW = 8;
                constexpr int CONCURRENT_ROWS = 8;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 64) {
                // in this case we would have threads_per_row = 32
                constexpr int THREADS_PER_ROW = 32;
                constexpr int CONCURRENT_ROWS = 2;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 128) {
                // in this case we would have threads_per_row = 64
                constexpr int THREADS_PER_ROW = 64;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 512) {
                // in this case we would have threads_per_row = 128
                constexpr int THREADS_PER_ROW = 128;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else {
                // in this case we would have threads_per_row = 256
                constexpr int THREADS_PER_ROW = 512;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            }

        } else if (num_rows <= NUM_SM * 32) {
            // in this case we would have iter=4
            constexpr int ITER_PER_BLOCK = 4;

            if (avg_elements_per_row < 16) {
                // in this case we would have threads_per_row = 16
                constexpr int THREADS_PER_ROW = 8;
                constexpr int CONCURRENT_ROWS = 8;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 64) {
                // in this case we would have threads_per_row = 32
                constexpr int THREADS_PER_ROW = 32;
                constexpr int CONCURRENT_ROWS = 2;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 128) {
                // in this case we would have threads_per_row = 64
                constexpr int THREADS_PER_ROW = 64;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 512) {
                // in this case we would have threads_per_row = 128
                constexpr int THREADS_PER_ROW = 128;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else {
                // in this case we would have threads_per_row = 256
                constexpr int THREADS_PER_ROW = 512;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            }

        } else {
            // in this case we would have iter=8
            constexpr int ITER_PER_BLOCK = 8;

            if (avg_elements_per_row < 16) {
                // in this case we would have threads_per_row = 16
                constexpr int THREADS_PER_ROW = 8;
                constexpr int CONCURRENT_ROWS = 8;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 64) {
                // in this case we would have threads_per_row = 32
                constexpr int THREADS_PER_ROW = 32;
                constexpr int CONCURRENT_ROWS = 2;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 128) {
                // in this case we would have threads_per_row = 64
                constexpr int THREADS_PER_ROW = 64;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else if (avg_elements_per_row < 512) {
                // in this case we would have threads_per_row = 128
                constexpr int THREADS_PER_ROW = 128;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            } else {
                // in this case we would have threads_per_row = 256
                constexpr int THREADS_PER_ROW = 512;
                constexpr int CONCURRENT_ROWS = 1;
                constexpr int BLOCK_SIZE = THREADS_PER_ROW * CONCURRENT_ROWS;
                index_t nblocks = (num_rows + CONCURRENT_ROWS * ITER_PER_BLOCK - 1) / (CONCURRENT_ROWS * ITER_PER_BLOCK);
                copyOutput<T, src_offset_t, THREADS_PER_ROW, CONCURRENT_ROWS, ITER_PER_BLOCK><<<nblocks, BLOCK_SIZE, 0, stream>>>(
                    dst, src,
                    offset_dst, offsets_src,
                    nums, num_rows,
                    copy_rows,
                    src_num_cols
                );
            }
        }

    }

    // Note that only 2 streams are used alternatively
    void outputCopyingLauncher(
        index_t* d_src_indices, data_t* d_src_values, index_t* d_src_row_offsets,
        index_t* d_dst_indices, data_t* d_dst_values, index_t* d_dst_row_offsets,
        index_t* h_num_rows,
        index_t* d_bins,
        index_t* d_num_elements,
        std::vector<cudaStream_t> &streams,
        index_t num_rows_a,
        index_t avg_elements_per_row,
        int copy_max_bin = 6
    ) {

        if (copy_max_bin < 0 || copy_max_bin > 6) {
            std::cout << "Invalid sort_max_bin value: " << copy_max_bin << std::endl;
            throw std::runtime_error("Invalid sort_max_bin value!");
        }

        if (h_num_rows[0] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[0], 
                d_bins + 0 * num_rows_a,
                avg_elements_per_row,
                streams[1]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[0],
                d_bins + 0 * num_rows_a,
                avg_elements_per_row,
                streams[2]
            );
        }
        CHECK_CUDA_KERNEL();
        
        if (copy_max_bin < 1) return;

        if (h_num_rows[1] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[1], 
                d_bins + 1 * num_rows_a,
                avg_elements_per_row,
                streams[3]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[1],
                d_bins + 1 * num_rows_a,
                avg_elements_per_row,
                streams[4]
            );
        }
        CHECK_CUDA_KERNEL();

        if (copy_max_bin < 2) return;

        if (h_num_rows[2] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[2], 
                d_bins + 2 * num_rows_a,
                avg_elements_per_row,
                streams[5]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[2],
                d_bins + 2 * num_rows_a,
                avg_elements_per_row,
                streams[6]
            );
        }
        CHECK_CUDA_KERNEL();

        if (copy_max_bin < 3) return;

        if (h_num_rows[3] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[3], 
                d_bins + 3 * num_rows_a,
                avg_elements_per_row,
                streams[7]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[3],
                d_bins + 3 * num_rows_a,
                avg_elements_per_row,
                streams[8]
            );
        }
        CHECK_CUDA_KERNEL();

        if (copy_max_bin < 4) return;

        if (h_num_rows[4] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[4], 
                d_bins + 4 * num_rows_a,
                avg_elements_per_row,
                streams[9]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[4],
                d_bins + 4 * num_rows_a,
                avg_elements_per_row,
                streams[10]
            );
        }
        CHECK_CUDA_KERNEL();

        if (copy_max_bin < 5) return;

        if (h_num_rows[5] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[5], 
                d_bins + 5 * num_rows_a,
                avg_elements_per_row,
                streams[11]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[5],
                d_bins + 5 * num_rows_a,
                avg_elements_per_row,
                streams[12]
            );
        }
        CHECK_CUDA_KERNEL();

        if (copy_max_bin < 6) return;

        if (h_num_rows[6] > 0) {
            copyKernelLauncher<index_t>(
                d_dst_indices, d_src_indices,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[6], 
                d_bins + 6 * num_rows_a,
                avg_elements_per_row,
                streams[13]
            );

            copyKernelLauncher<data_t>(
                d_dst_values, d_src_values,
                d_dst_row_offsets, d_src_row_offsets,
                d_num_elements, h_num_rows[6],
                d_bins + 6 * num_rows_a,
                avg_elements_per_row,
                streams[14]
            );
        }

        CHECK_CUDA_KERNEL();
    }



}