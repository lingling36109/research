#pragma once
#include <vector>

#include "Analysis.cuh"
#include "AccumulatorESC.cuh"
#include "AccumulatorHash.cuh"
#include "AccumulatorDense.cuh"
#include "MurmurHash.cuh"
#include "HLL.cuh"
#include "Epilogue.cuh"
#include "Wrappers.cuh"
#include "Segmentation.cuh"
#include "CSR.h"
#include "Utils.h"


namespace somespgemm {


// workspace for related vars
// gpu vars will start as "d_"
// those without "d_" can be seen as host vars
struct Workspace {
    std::shared_ptr<cuCSR> A;
    std::shared_ptr<cuCSR> B;
    std::shared_ptr<cuCSR> C;

    std::vector<cudaStream_t> streams;
    const int NSTREAMS = 20;

    std::vector<cudaEvent_t> events;
    const int NEVENTS = 200;

    const int num_sections = 1; // not supported now

    index_t ana1_avg_product = 0;
    int ana1_type = 0;  // 0: Ultra Sparse (ESC); 1: Others
    double ana2_avg_sampled_compaction = 0.0;
    double ana2_avg_sampled_compaction2 = 0.0;
    int ana2_type = 0;                      // 0: Low Compaction (Go to Symbolic); 1: Average or High Compaction (Skip Symbolic)
    bool dense_query_bitmap = false;

    // prefix sums and total sums of *products* are stored using size_t to avoid overflow
    index_t *d_num_products;               
    size_t  *d_num_products_prefixsum;  // used by ESC kernel    
    index_t *d_max_product;
    index_t h_max_product;
    size_t total_products;

    index_t max_a_row_nnz;  // This is used for load balance of HLL Merge
    index_t avg_a_row_nnz;
    index_t* d_max_a_row_nnz;
    bool use_multiple_merge_kernels;

    index_t *d_max_b_row_len;
    index_t *d_avg_b_row_len;
    index_t *d_b_col_idx_left;
    index_t *d_b_col_idx_right;

    index_t *d_total_sample_rows_nnz;
    index_t *d_total_sampled_products;
    index_t *d_num_early_exit_rows;
    double *d_sampled_compaction_sum;
    index_t h_total_sample_rows_nnz;
    index_t h_num_sampled_products;
    index_t h_num_early_exit_rows;
    int early_exit_threshold = 0;    // break threshold for sampling
    int num_sample_rows = 0;              // number of sampled rows
    double* d_sampled_compaction;
    std::vector<double> h_sampled_compaction;
    double compaction_avg = 0.0;
    double compaction_var = 0.0;
    double safe_estimated_avg_compaction = 0.0;
    bool use_estimated_symbolic = false;

    bool spark_need_outlier = false;
    int spark_kernel_size;
    int spark_outlier_kernel_size;
    index_t* d_num_outlier_rows;
    index_t* d_outlier_rows;
    index_t h_num_outlier_rows;

    // for cub ops
    void *d_temp_storage;
    size_t temp_storage_bytes;
    // index_t global_mem_buffer_size;
    index_t* d_global_mem_buffer_0;
    uint32_t* d_global_mem_bitmap_0;

    index_t* d_global_mem_buffer_1;
    index_t* d_global_mem_bitmap_1;

    index_t* d_global_mem_buffer_2;
    index_t* d_global_mem_bitmap_2;

    data_t* d_global_mem_buffer_3;
    index_t* d_global_mem_bitmap_3;

    // for hll
    int hll_log2_precision = 0;
    uint8_t *d_hll_b;
    uint8_t *d_hll_c;
    index_t *d_est;
    index_t *d_psum_est;
    size_t total_est;   // this is currently using size_t because we have a expansion factor 
    double hll_expansion_factor = 0.0;

    // the bins are used in multiple places
    index_t* d_bin_count;
    index_t* d_bins;
    std::vector<index_t> h_bin_count;
    bool hash_numeric_use_largest = false;

    // for est kernel output
    index_t *d_estik_indices;
    data_t  *d_estik_values;

    index_t *d_overflow_rows;
    index_t *d_num_overflow_rows;
    index_t h_num_overflow_rows;

    index_t* d_overflow_buffer_allocated;
    index_t h_overflow_buffer_allocated;
    index_t* d_hybrid_hashmap_cnt;
    index_t hybrid_hashmap_max;

    // output related vars
    index_t *d_num_outputs_row;
    index_t *d_psum_outputs_row; // this one should NOT be freed since may be owned by C
    index_t total_outputs;

    std::vector<index_t> bin1;
    std::vector<index_t> bin2;

    // Segmentation (spy-plot image segmentation) result
    SegmentationResult segmentation;
    bool segmentation_ran = false;



    Workspace(std::shared_ptr<cuCSR> A, std::shared_ptr<cuCSR> B)
        : A(A), B(B), C(std::make_shared<cuCSR>()), 
        spark_need_outlier(false), spark_kernel_size(0), spark_outlier_kernel_size(0),
        d_num_products(nullptr), d_num_products_prefixsum(nullptr), 
        d_max_a_row_nnz(nullptr), max_a_row_nnz(0), avg_a_row_nnz(0), use_multiple_merge_kernels(false),
        d_max_b_row_len(nullptr), d_avg_b_row_len(nullptr), d_b_col_idx_left(nullptr), d_b_col_idx_right(nullptr),
        d_temp_storage(nullptr), temp_storage_bytes(0), 
        d_global_mem_buffer_0(nullptr), d_global_mem_bitmap_0(nullptr),
        d_global_mem_buffer_1(nullptr), d_global_mem_bitmap_1(nullptr),
        d_global_mem_buffer_2(nullptr), d_global_mem_bitmap_2(nullptr),
        d_global_mem_buffer_3(nullptr), d_global_mem_bitmap_3(nullptr),
        d_num_outputs_row(nullptr), d_psum_outputs_row(nullptr),
        d_hll_b(nullptr), d_hll_c(nullptr), d_est(nullptr),
        d_bin_count(nullptr), d_bins(nullptr),
        d_psum_est(nullptr), d_hybrid_hashmap_cnt(nullptr),
        d_estik_indices(nullptr), d_estik_values(nullptr), 
        d_num_overflow_rows(nullptr), d_overflow_rows(nullptr), h_num_overflow_rows(0),
        d_num_outlier_rows(nullptr), d_outlier_rows(nullptr), h_num_outlier_rows(0),
        d_max_product(nullptr), d_total_sample_rows_nnz(nullptr), d_num_early_exit_rows(nullptr), d_sampled_compaction(nullptr),
        d_total_sampled_products(nullptr), d_sampled_compaction_sum(nullptr),
        d_overflow_buffer_allocated(nullptr),
        h_max_product(0), h_total_sample_rows_nnz(0), h_num_early_exit_rows(0),
        h_overflow_buffer_allocated(0)
        {
            streams.resize(NSTREAMS);
            for (int i = 0; i < NSTREAMS; ++i) {
                CHECK_CUDA(cudaStreamCreate(&streams[i]));
            }
            events.resize(NEVENTS);
            for (int i = 0; i < NEVENTS; ++i) {
                CHECK_CUDA(cudaEventCreate(&events[i]));
            }
        }

    ~Workspace() {
        for (auto& s : streams) {
            cudaStreamDestroy(s);
        }
        for (auto& e : events) {
            cudaEventDestroy(e);
        }
    }

    void syncMainToStreams(int num_streams, int sync_id) {
        if (sync_id * NSTREAMS >= NEVENTS) {
            throw std::invalid_argument("Not enough events to sync streams");
        }
        for (int i = 1; i < num_streams; ++i) {
            cudaEventRecord(events[sync_id * NSTREAMS + i], streams[i]);
            cudaStreamWaitEvent(streams[0], events[sync_id * NSTREAMS + i], 0);
            CHECK_CUDA_KERNEL();
        }
    }

    void syncStreamsToMain(int num_streams, int sync_id) {
        if (sync_id * NSTREAMS >= NEVENTS) {
            throw std::invalid_argument("Not enough events to sync streams");
        }
        for (int i = 1; i < num_streams; ++i) {
            // one event should be enough
            cudaEventRecord(events[sync_id * NSTREAMS + i], streams[0]);
            cudaStreamWaitEvent(streams[i], events[sync_id * NSTREAMS + i], 0);
            CHECK_CUDA_KERNEL();
        }
    }

};

class SpGEMM {
public:
    SpGEMM(const Config& config) : config_(config) {}

    std::shared_ptr<cuCSR> run(std::shared_ptr<cuCSR> A, std::shared_ptr<cuCSR> B) {

        cuTimer timer;
        Workspace workspace(A, B);
        Timing timing;

        cudaMemPool_t mempool;
        auto device = 0;
        cudaDeviceGetDefaultMemPool(&mempool, device);
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

        double time = 0.0;
        for (int i = 0; i < config_.warmup_iters + config_.bench_iters; ++i) {
            if (i == config_.warmup_iters) {
                timing.reset();
            }

            workspace.C->reset(workspace.streams[0]);

            cudaDeviceSynchronize();
            timer.startTimer();

            prologue(workspace, config_, timing);
            
            analysis(workspace, config_, timing);

            if (config_.use_segmented_workflow && workspace.ana1_type != 2) {
                // Spy-plot image segmentation (GPU). Produces a per-row-range
                // segment map with classification labels. Currently used as an
                // analysis-only/instrumentation pass; per-segment kernel
                // dispatch is a follow-up. The segment map and stats are
                // emitted to the stats JSON when output_stats is enabled.
                runSegmentation(
                    workspace.A->row_offsets, workspace.A->col_ids,
                    workspace.A->rows, workspace.A->nnz, workspace.B->cols,
                    workspace.d_num_products,
                    workspace.d_max_b_row_len,
                    workspace.d_avg_b_row_len,
                    workspace.d_b_col_idx_left,
                    workspace.d_b_col_idx_right,
                    config_,
                    workspace.streams[0],
                    workspace.segmentation
                );
                workspace.segmentation_ran = true;
            }

            if (workspace.ana1_type == 1) {
                if (workspace.hll_log2_precision == 5)
                    sample_kernel<5>(workspace, config_, timing);
                else if (workspace.hll_log2_precision == 6)
                    sample_kernel<6>(workspace, config_, timing);
                else 
                    throw std::invalid_argument("Unsupported HLL precision");
            }

            if (workspace.ana1_type == 1 && workspace.ana2_type == 0) {
                symbolic_kernel(workspace, config_, timing);
            }

            if (workspace.ana1_type == 1 && workspace.ana2_type == 1) {
                if (workspace.hll_log2_precision == 5)
                    est_kernel<5>(workspace, config_, timing);
                else if (workspace.hll_log2_precision == 6)
                    est_kernel<6>(workspace, config_, timing);
                else 
                    throw std::invalid_argument("Unsupported HLL precision");
            }
            
            if (workspace.ana1_type == 0) {
                numeric_kernel_ultrasparse(workspace, config_, timing);
            } else if (workspace.ana1_type == 2) {
                numeric_kernel_fat(workspace, config_, timing);
            } else if (workspace.ana2_type == 0) {
                numeric_kernel_precise(workspace, config_, timing);
            } else {
                numeric_kernel_est(workspace, config_, timing);
            }

            timer.stopTimer();

            int matrix_type = 0;
            if (workspace.ana1_type == 0) {
                matrix_type = 0;
            } else if (workspace.ana1_type == 2) {
                matrix_type = 3;
            } else if (workspace.ana2_type == 0) {
                matrix_type = 1;
            } else {
                matrix_type = 2;
            }

#ifndef QUIET
            std::cout << "Iteration " << i << " done. Time: " << timer.getElapsedTime() << " ms\n";
            std::cout << "Matrix type: " << matrix_type << std::endl;
#endif

            if (i >= config_.warmup_iters) {
                time += timer.getElapsedTime();
            }
            if (i != config_.warmup_iters + config_.bench_iters - 1) {
                deallocate(workspace, config_, timing);
            }
            if (config_.print_memory) {
                printMemInfo();
            }
        }
        
        std::cout << "Average SpGEMM time over " << config_.bench_iters << " iters: " 
                  << (time / config_.bench_iters) << " ms\n";

        if (config_.track_stage_times) {
            timing.report(config_.bench_iters, workspace.total_products * 2, time / config_.bench_iters);
        }

        if (config_.output_stats) {
            stats_ = save_workspace_to_json(workspace, config_);
            if (config_.track_stage_times) {
                stats_["timing"] = timing.save_to_json(config_.bench_iters);
            }
            stats_["total_time"] = time / config_.bench_iters;
            stats_["gflops"] = (workspace.total_products * 2) / (time / config_.bench_iters) / 1e6;
        }

        return workspace.C;
    }

    json get_stats() const {
        return stats_;
    }

private:
    Config config_;
    json stats_;


    void prologue(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer();
        }

        if (config.use_hybrid_accumulator && (!config.use_indirect_sort)) {
            std::cerr << "Hybrid accumulator is enabled, but indirect sort is disabled.\n";
            throw std::invalid_argument("Invalid configuration");
        }

        CHECK_CUDA(cudaMallocAsync(&workspace.d_num_products, sizeof(index_t) * (workspace.A->rows + 1), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_num_products_prefixsum, sizeof(size_t) * (workspace.A->rows + 1), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_max_b_row_len, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_avg_b_row_len, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_b_col_idx_left, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_b_col_idx_right, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_num_outputs_row, sizeof(index_t) * (workspace.A->rows + 1), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_max_product, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_max_product, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_max_a_row_nnz, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_max_a_row_nnz, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_total_sample_rows_nnz, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_total_sample_rows_nnz, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_total_sampled_products, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_total_sampled_products, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_sampled_compaction_sum, sizeof(double), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_sampled_compaction_sum, 0, sizeof(double), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_num_early_exit_rows, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_num_early_exit_rows, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_num_overflow_rows, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_num_overflow_rows, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_overflow_buffer_allocated, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_overflow_buffer_allocated, 0, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_hybrid_hashmap_cnt, sizeof(index_t), workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_hybrid_hashmap_cnt, 0, sizeof(index_t), workspace.streams[0]));

        if (workspace.C->row_offsets != nullptr) {
            workspace.d_psum_outputs_row = workspace.C->row_offsets;
        } else {
            CHECK_CUDA(cudaMallocAsync(&workspace.d_psum_outputs_row, sizeof(index_t) * (workspace.A->rows + 1), workspace.streams[0]));
        }
        // get temp byte size
        size_t temp_bytes = 0;
        // TODO: make sure this would not cause a problem! the num product is actually smaller 
        workspace.d_temp_storage = nullptr;
        workspace.hybrid_hashmap_max = workspace.A->rows / 10;

        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage, temp_bytes,
            workspace.d_num_products, workspace.d_num_products_prefixsum, workspace.A->rows,
            workspace.streams[0]
        ));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        workspace.temp_storage_bytes = temp_bytes;

        temp_bytes = 0;
        CHECK_CUDA(cub::DeviceReduce::Max(
            workspace.d_temp_storage, temp_bytes,
            workspace.d_num_products, workspace.d_num_outputs_row, workspace.A->rows,
            workspace.streams[0]
        ));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        workspace.temp_storage_bytes = std::max(workspace.temp_storage_bytes, temp_bytes);

        temp_bytes = 0;
        CHECK_CUDA(cub::DeviceReduce::Sum(
            workspace.d_temp_storage, temp_bytes,
            workspace.d_num_products, workspace.d_num_outputs_row, workspace.A->rows,
            workspace.streams[0]
        ));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        workspace.temp_storage_bytes = std::max(workspace.temp_storage_bytes, temp_bytes);

        CHECK_CUDA(cudaMallocAsync(&workspace.d_temp_storage, workspace.temp_storage_bytes, workspace.streams[0]));
        if (config.track_stage_times) {
            timer.stopTimer();
            timing.prologue += timer.getElapsedTime();
        }

        int idx_bits_needed = bits_needed(workspace.B->cols-1);
        if (idx_bits_needed > HASH_NUMERIC_IDX_BITS[6] || !config.use_hybrid_accumulator) {
            workspace.hash_numeric_use_largest = false;
        } else {
            if (workspace.B->cols > config.numeric_hybrid_hash_kernel_row_length_threshold) {
                workspace.hash_numeric_use_largest = true;
            }
        }
    }


    void analysis(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer(workspace.streams[0]);
        }

        // Ana1: Analyze row product for shortcut to ESC/FAT and skipping HLL
        if (workspace.A->cols / config.fat_ratio > workspace.A->rows &&
            workspace.B->rows / config.fat_ratio > workspace.B->cols &&
            workspace.B->cols < DENSE_NUMERIC_BIN_SIZES[5] &&
            config.use_fat_workflow
        ) {
            // Shortcut to FAT
            // This is a special case for extremely fat, short matrix
            workspace.ana1_type = 2;
            return;
        }

        analysisKernelWrapper(
            workspace.A->row_offsets, workspace.A->col_ids, workspace.A->rows, workspace.A->nnz, 
            workspace.B->row_offsets, workspace.B->col_ids, workspace.B->rows, workspace.B->nnz, 
            workspace.d_num_products,
            workspace.d_max_b_row_len,
            workspace.d_avg_b_row_len,
            workspace.d_b_col_idx_left,
            workspace.d_b_col_idx_right,
            workspace.d_max_a_row_nnz,
            workspace.streams[0]
        );
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.analysis_product_calc += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        // This should not overflow size_t theoretically.
        // However seems not to be the case in practice.
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage, workspace.temp_storage_bytes,
            workspace.d_num_products, workspace.d_num_products_prefixsum, workspace.A->rows,
            workspace.streams[0]
        ));

        CHECK_CUDA(cub::DeviceReduce::Max(
            workspace.d_temp_storage, workspace.temp_storage_bytes,
            workspace.d_num_products, workspace.d_max_product, workspace.A->rows,
            workspace.streams[0]
        ));

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.analysis_reduce += timer.getElapsedTime();
            timer.startTimer();
        }


        size_t last_prefix_sum = 0;
        index_t last_item_num = 0;
        CHECK_CUDA(cudaMemcpyAsync(&last_prefix_sum, workspace.d_num_products_prefixsum + workspace.A->rows - 1, sizeof(size_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&last_item_num, workspace.d_num_products + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&workspace.h_max_product, workspace.d_max_product, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&workspace.max_a_row_nnz, workspace.d_max_a_row_nnz, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));

        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        workspace.total_products = last_prefix_sum + last_item_num;
        auto avg_product = (double)workspace.total_products / workspace.A->rows;
        workspace.ana1_avg_product = (index_t)avg_product;

        if (avg_product <= (double)config.usparse_avg_product_threshold && config.use_ultrasparse_workflow) {
            workspace.ana1_type = 0;
            for (auto k : NUMERIC_USPARSE_KERNEL_SIZES)
                if (k >= avg_product) {
                    workspace.spark_kernel_size = k;
                    break;
                }
            if ((workspace.h_max_product) > workspace.spark_kernel_size) {
                workspace.spark_need_outlier = true;
                workspace.spark_outlier_kernel_size = (((workspace.h_max_product) + 31) / 32) * 32;
            }
        } else {
            auto ratio = workspace.total_products / workspace.A->nnz;
            workspace.ana1_type = 1;
            if (ratio < config.hll_higher_precision_expansion_threshold) {
                workspace.hll_log2_precision = config.hll_precision_low;
                workspace.hll_expansion_factor = config.hll_expansion_coes.at(config.hll_precision_low);
            } else {
                workspace.hll_log2_precision = config.hll_precision_high;
                workspace.hll_expansion_factor = config.hll_expansion_coes.at(config.hll_precision_high);
            }
        }

        if (config.track_stage_times) {
            timer.stopTimer();
            timing.analysis_mem_cpy += timer.getElapsedTime();
        }

    }

    template<int LOG2_PRECISION>
    void sample_kernel(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        constexpr int PRECISION = 1 << LOG2_PRECISION;
        cuTimer timer;

        if (config.track_stage_times) {
            timer.startTimer(workspace.streams[0]);
        }

        // allocate HLL related memory
        size_t alloc_size = sizeof(uint8_t) * PRECISION * workspace.num_sections * (workspace.B->rows);
        CHECK_CUDA(cudaMallocAsync(&workspace.d_hll_b, alloc_size, workspace.streams[0]));
        alloc_size = sizeof(uint8_t) * PRECISION * workspace.num_sections * (workspace.A->rows);
        CHECK_CUDA(cudaMallocAsync(&workspace.d_hll_c, alloc_size, workspace.streams[0]));

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.est_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }
        
        // Matrix B HLL construct
        int rows_per_block = workspace.B->rows / HLL_K_CONSTRUCT_NBLOCKS;
        if (rows_per_block < HLL_K_CONSTRUCT_MIN_ROWS_PER_BLOCK) rows_per_block = HLL_K_CONSTRUCT_MIN_ROWS_PER_BLOCK;
        if (rows_per_block > HLL_K_CONSTRUCT_MAX_ROWS_PER_BLOCK) rows_per_block = HLL_K_CONSTRUCT_MAX_ROWS_PER_BLOCK;
        size_t nblocks = (workspace.B->rows + rows_per_block - 1) / rows_per_block;
        size_t shared_mem_size = sizeof(uint32_t) * PRECISION * workspace.num_sections * rows_per_block + (1 + rows_per_block) * sizeof(index_t);
        hllConstruct<LOG2_PRECISION><<<nblocks, 32, shared_mem_size, workspace.streams[0]>>>(
            workspace.B->row_offsets, workspace.B->col_ids, workspace.B->rows,
            workspace.B->cols, workspace.B->nnz,
            workspace.d_hll_b, workspace.num_sections, rows_per_block
        );
        CHECK_CUDA_KERNEL();
        
        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.est_hll_construct += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        int hll_rows = (int)(workspace.A->rows * config.hll_sample_rate);
        if (hll_rows < config.hll_sample_min_rows) hll_rows = config.hll_sample_min_rows;
        if (hll_rows > config.hll_sample_max_rows) hll_rows = config.hll_sample_max_rows;
        workspace.num_sample_rows = hll_rows;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_sampled_compaction, sizeof(double) * hll_rows, workspace.streams[0]));

        shared_mem_size = 32 * HLL_SAMPLE_K_THREADS_PER_BLOCK;
        hllSamplingMerge<LOG2_PRECISION><<<hll_rows, HLL_SAMPLE_K_THREADS_PER_BLOCK, shared_mem_size, workspace.streams[0]>>>
        (
            workspace.A->row_offsets, workspace.A->col_ids, workspace.A->rows, workspace.A->nnz,
            workspace.d_hll_b,
            workspace.d_num_products, 
            workspace.d_total_sample_rows_nnz, workspace.d_total_sampled_products,
            workspace.d_sampled_compaction_sum, workspace.d_sampled_compaction
        );
        workspace.h_sampled_compaction.resize(hll_rows);
        CHECK_CUDA_KERNEL();

        CHECK_CUDA(cudaMemcpyAsync(&workspace.h_total_sample_rows_nnz, workspace.d_total_sample_rows_nnz, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&workspace.h_num_sampled_products, workspace.d_total_sampled_products, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&workspace.ana2_avg_sampled_compaction2, workspace.d_sampled_compaction_sum, sizeof(double), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(workspace.h_sampled_compaction.data(), workspace.d_sampled_compaction, sizeof(double) * hll_rows, cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        workspace.ana2_avg_sampled_compaction = (double)workspace.h_num_sampled_products / (double)workspace.h_total_sample_rows_nnz;
        workspace.ana2_avg_sampled_compaction2 /= hll_rows;

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.est_sample += timer.getElapsedTime();
        }

        // decide matrix type
        workspace.ana2_type = decideMatrixType(
            workspace.A->nnz,
            workspace.total_products,
            workspace.ana2_avg_sampled_compaction,
            workspace.ana2_avg_sampled_compaction2,
            config
        );

        if (!config.use_estimation_workflow) {
            workspace.ana2_type = 0;
        }

        workspace.avg_a_row_nnz = workspace.A->nnz / workspace.A->rows;

        if ((workspace.ana2_avg_sampled_compaction < config.dense_query_bitmap_compaction_threshold && \
            workspace.ana2_avg_sampled_compaction2 < config.dense_query_bitmap_compaction_threshold) || \
            (!config.use_assist_information)
        ) {
            workspace.dense_query_bitmap = false;
        } else {
            workspace.dense_query_bitmap = true;
        }

        if(workspace.ana2_type == 1) {
            return;
        }

        if (config.track_stage_times) {
            timer.startTimer();
        }

        // Host calculate of compaction var and avg
        double compaction_sum = 0.0;
        for (int i = 0; i < hll_rows; ++i) {
            compaction_sum += workspace.h_sampled_compaction[i];
        }
        workspace.compaction_avg = compaction_sum / hll_rows;
        double compaction_var_sum = 0.0;
        for (int i = 0; i < hll_rows; ++i) {
            compaction_var_sum += (workspace.h_sampled_compaction[i] - workspace.compaction_avg) * (workspace.h_sampled_compaction[i] - workspace.compaction_avg);
        }
        workspace.compaction_var = compaction_var_sum / (hll_rows-1);

        // double expansion = 1.0 + config.target_z_value * sqrt(workspace.compaction_var) / workspace.compaction_avg;
        workspace.safe_estimated_avg_compaction = workspace.compaction_avg - config.estimate_compaction_target_z_value * sqrt(workspace.compaction_var);
        workspace.safe_estimated_avg_compaction /= 2;       // Do not over compact!
        if (workspace.safe_estimated_avg_compaction > workspace.compaction_avg * 0.8) {
            workspace.safe_estimated_avg_compaction = workspace.compaction_avg * 0.8;
            // make up for the hashmap load factor when we're too confident
        }

        if (config.track_stage_times) {
            timer.stopTimer();
            timing.sampling_var_calc_host += timer.getElapsedTime();
        }

        if (workspace.safe_estimated_avg_compaction >= 1.5 && config.use_assist_information) {
            workspace.use_estimated_symbolic = true;
        } else {
            workspace.use_estimated_symbolic = false;
        }

        CHECK_CUDA_KERNEL();

    }

    template<int LOG2_PRECISION>
    void est_kernel(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        constexpr int PRECISION = 1 << LOG2_PRECISION;
        cuTimer timer;

        if (config.track_stage_times) {
            timer.startTimer(workspace.streams[0]);
        }

        size_t alloc_size = sizeof(index_t) * (workspace.A->rows) * workspace.num_sections;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_est, alloc_size, workspace.streams[0]));
        alloc_size = sizeof(index_t) * (workspace.A->rows) * workspace.num_sections;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_psum_est, alloc_size, workspace.streams[0]));

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.est_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        if (workspace.max_a_row_nnz / workspace.avg_a_row_nnz > HLL_MERGE_MULTI_KERNEL_THRESHOLD) {
            workspace.syncStreamsToMain(2, 7);

            workspace.use_multiple_merge_kernels = true;
            size_t nblocks = workspace.A->rows;
            size_t shared_mem_size = sizeof(uint32_t) * 64 * HLL_MERGE_MULTI_KERNEL_THRESHOLD * HLL_MERGE_K_BYTES_PER_THREAD;
            hllMerge<LOG2_PRECISION, false, true><<<nblocks, 64 * HLL_MERGE_MULTI_KERNEL_THRESHOLD, shared_mem_size, workspace.streams[1]>>>(
                workspace.A->row_offsets, workspace.A->col_ids, workspace.A->rows,
                workspace.A->cols, workspace.A->nnz,
                workspace.d_hll_b, workspace.d_hll_c, workspace.d_est, 
                workspace.hll_expansion_factor,
                workspace.avg_a_row_nnz * 4
            );

            shared_mem_size = sizeof(uint32_t) * 64 * HLL_MERGE_K_BYTES_PER_THREAD;
            hllMerge<LOG2_PRECISION, true, false><<<nblocks, 64, shared_mem_size, workspace.streams[0]>>>(
                workspace.A->row_offsets, workspace.A->col_ids, workspace.A->rows,
                workspace.A->cols, workspace.A->nnz,
                workspace.d_hll_b, workspace.d_hll_c, workspace.d_est, 
                workspace.hll_expansion_factor,
                workspace.avg_a_row_nnz * 4
            );
            workspace.syncMainToStreams(2, 8);

        } else {
            // Merge into Matrix C HLL
            size_t nblocks = workspace.A->rows;
            size_t shared_mem_size = sizeof(uint32_t) * 64 * HLL_MERGE_K_BYTES_PER_THREAD;
            hllMerge<LOG2_PRECISION><<<nblocks, 64, shared_mem_size, workspace.streams[0]>>>(
                workspace.A->row_offsets, workspace.A->col_ids, workspace.A->rows,
                workspace.A->cols, workspace.A->nnz,
                workspace.d_hll_b, workspace.d_hll_c, workspace.d_est,
                workspace.hll_expansion_factor
            );
        }
        CHECK_CUDA_KERNEL();
        
        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.est_hll_merge += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }
    }

    void symbolic_kernel(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {

        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer();
        }

        CHECK_CUDA(cudaMallocAsync(&workspace.d_bin_count, sizeof(index_t) * BIN_NUM * 2, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_bins, sizeof(index_t) * workspace.A->rows * BIN_NUM * 2, workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_bin_count, 0, sizeof(index_t) * BIN_NUM * 2, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_overflow_rows, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
        index_t* bin_count = workspace.d_bin_count;
        index_t* bins = workspace.d_bins;

        const int items_per_block = BINNING_K_ITER_PER_BLOCK * BINNING_K_BLOCK_SIZE;
        int nblocks = (workspace.A->rows + items_per_block - 1) / items_per_block;

        if (workspace.use_estimated_symbolic) {
            binning<true, true><<<nblocks, BINNING_K_BLOCK_SIZE, 0, workspace.streams[0]>>>(
                workspace.d_num_products, 
                bin_count, 
                bins,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.A->rows, 
                items_per_block,
                config.symbolic_expansion_coe,
                workspace.safe_estimated_avg_compaction
            );
        } else {
            binning<true, false><<<nblocks, BINNING_K_BLOCK_SIZE, 0, workspace.streams[0]>>>(
                workspace.d_num_products, 
                bin_count, 
                bins,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.A->rows, 
                items_per_block,
                config.symbolic_expansion_coe
            );
        }

        CHECK_CUDA_KERNEL();

        workspace.h_bin_count.resize(BIN_NUM * 2);
        index_t* h_bin_count = workspace.h_bin_count.data();
        CHECK_CUDA(cudaMemcpyAsync(h_bin_count, bin_count, sizeof(index_t) * (BIN_NUM * 2), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));

        if (config.print_info) {
            workspace.bin1 = workspace.h_bin_count;
        }

        CHECK_CUDA_KERNEL();

        index_t* matC_nnz_per_row = workspace.d_num_outputs_row;

        // symbolic kernel launch
        if (config.track_stage_times) {
            timer.stopTimer();
            timing.symbolic_binning += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }       

        int overflow_buffer_count = workspace.A->rows / 100;

        if (workspace.use_estimated_symbolic) {
            overflow_buffer_count = (overflow_buffer_count + 31) / 32 * 32;

            CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_bitmap_2, overflow_buffer_count / 8, workspace.streams[0]));
            CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_bitmap_2, 0, overflow_buffer_count / 8, workspace.streams[0]));
            CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_buffer_2, sizeof(index_t) * overflow_buffer_count * (workspace.h_max_product + 6), workspace.streams[0]));
            CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_buffer_2, 0xFF, sizeof(index_t) * overflow_buffer_count * (workspace.h_max_product + 6), workspace.streams[0]));
        }

        workspace.syncStreamsToMain(BIN_NUM*2, 1);
        
        if (workspace.use_estimated_symbolic) {
            hashSymbolicLauncher<true>(
                workspace.A,
                workspace.B,
                workspace.h_bin_count,
                workspace.d_bins,
                workspace.d_num_products,
                workspace.d_max_b_row_len,
                workspace.h_max_product,
                workspace.d_global_mem_buffer_0,
                workspace.d_num_outputs_row,
                workspace.ana2_avg_sampled_compaction,
                !config.use_assist_information,
                workspace.streams,
                workspace.d_global_mem_buffer_2,
                workspace.d_global_mem_bitmap_2,
                overflow_buffer_count,
                workspace.h_max_product + 6
            );
        } else {
            hashSymbolicLauncher<false>(
                workspace.A,
                workspace.B,
                workspace.h_bin_count,
                workspace.d_bins,
                workspace.d_num_products,
                workspace.d_max_b_row_len,
                workspace.h_max_product,
                workspace.d_global_mem_buffer_0,
                workspace.d_num_outputs_row,
                workspace.ana2_avg_sampled_compaction,
                !config.use_assist_information,
                workspace.streams
            );
        }

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.symbolic_hash += timer.getElapsedTime(); 
            timer.startTimer(workspace.streams);       
        }

        if (workspace.dense_query_bitmap) {
            denseSymbolicLauncher<true>(
                workspace.A,
                workspace.B,
                workspace.h_bin_count.data() + BIN_NUM,
                workspace.d_bins + BIN_NUM * workspace.A->rows,
                workspace.d_num_products,
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.h_max_product,
                workspace.d_num_outputs_row,
                workspace.ana2_avg_sampled_compaction,
                workspace.streams.data() + BIN_NUM
            );
        } else {
            denseSymbolicLauncher<false>(
                workspace.A,
                workspace.B,
                workspace.h_bin_count.data() + BIN_NUM,
                workspace.d_bins + BIN_NUM * workspace.A->rows,
                workspace.d_num_products,
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.h_max_product,
                workspace.d_num_outputs_row,
                workspace.ana2_avg_sampled_compaction,
                workspace.streams.data() + BIN_NUM
            );
        }

        workspace.syncMainToStreams(BIN_NUM*2, 2);

        
        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.symbolic_dense += timer.getElapsedTime();        
        }
        
        return;
    }

    void numeric_kernel_fat(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer();
        }
        size_t output_matrix_size = workspace.A->rows * workspace.B->cols;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_estik_indices, sizeof(index_t) * output_matrix_size, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_estik_values, sizeof(data_t) * output_matrix_size, workspace.streams[0]));

        if (config.track_stage_times) {
            timer.stopTimer();
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        // select a dense bin
        int bin = BIN_NUM-1;
        int concurrent_blocks = 0;
        for (int i = 1; i < BIN_NUM-1; i++) {
            if (workspace.B->cols < DENSE_NUMERIC_BIN_SIZES[i]) {
                bin = i;
                concurrent_blocks = CONCURRENT_BLOCKS_PER_SM[i];
                break;
            }
        }

        while (bin < BIN_NUM-2 && \
            workspace.A->rows < concurrent_blocks * NUM_SM * 2) {
            bin++;
            concurrent_blocks = CONCURRENT_BLOCKS_PER_SM[bin];
        }

        double average_nnz_per_b_row = static_cast<double>(workspace.B->nnz) / workspace.B->rows;
        int log_nthr_per_a_elem = 0;
        while ((1 << log_nthr_per_a_elem) < average_nnz_per_b_row) {
            log_nthr_per_a_elem++;
        }

        denseNumericStaticLauncher(
            workspace.A,
            workspace.B,
            workspace.d_estik_indices,
            workspace.d_estik_values,
            log_nthr_per_a_elem,
            bin,
            workspace.streams[0],
            workspace.d_num_outputs_row
        );

        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage,
            workspace.temp_storage_bytes,
            workspace.d_num_outputs_row,
            workspace.d_psum_outputs_row,
            workspace.A->rows,
            workspace.streams[0]
        ));

        if (config.track_stage_times) {
            timer.stopTimer();
            timing.numeric_fat += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        size_t C_nnz = 0;
        index_t last_prefix_sum = 0;
        index_t last_item_num = 0;
        CHECK_CUDA(cudaMemcpyAsync(&last_prefix_sum, workspace.d_psum_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&last_item_num, workspace.d_num_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        C_nnz = last_prefix_sum + last_item_num;

        if (workspace.C->nnz == 0) {
            workspace.C->alloc(workspace.A->rows, workspace.B->cols, C_nnz, false, workspace.streams[0]);
        }
        workspace.C->row_offsets = workspace.d_psum_outputs_row;
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) { 
            timer.stopTimer();
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        // calculate the row offsets
        workspace.syncStreamsToMain(3, 5);

        index_t avg_elements_per_row = workspace.C->nnz / workspace.A->rows;

        copyKernelLauncher<index_t, size_t>(
            workspace.C->col_ids, workspace.d_estik_indices,
            workspace.C->row_offsets, nullptr,
            workspace.d_num_outputs_row,
            workspace.A->rows,
            nullptr,
            avg_elements_per_row,
            workspace.streams[1],
            workspace.B->cols
        );

        copyKernelLauncher<data_t, size_t>(
            workspace.C->data, workspace.d_estik_values,
            workspace.C->row_offsets, nullptr,
            workspace.d_num_outputs_row,
            workspace.A->rows,
            nullptr,
            avg_elements_per_row,
            workspace.streams[0],
            workspace.B->cols
        );

        workspace.syncMainToStreams(3, 6);

        CHECK_CUDA_KERNEL();
        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.epilogue_copy += timer.getElapsedTime();
        }
        return;


    }

    void numeric_kernel_ultrasparse(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer();
        }
        CHECK_CUDA(cudaMallocAsync(&workspace.d_estik_indices, sizeof(index_t) * workspace.total_products, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_estik_values, sizeof(data_t) * workspace.total_products, workspace.streams[0]));

        if (config.track_stage_times) {
            timer.stopTimer();
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        workspace.syncStreamsToMain(2, 3);

        if (workspace.spark_need_outlier) {
            index_t* &num_outlier_rows = workspace.d_num_outlier_rows;
            index_t* &outlier_rows = workspace.d_outlier_rows;
            CHECK_CUDA(cudaMallocAsync(&num_outlier_rows, sizeof(index_t), workspace.streams[0]));
            CHECK_CUDA(cudaMallocAsync(&outlier_rows, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
            CHECK_CUDA(cudaMemsetAsync(num_outlier_rows, 0, sizeof(index_t), workspace.streams[0]));
            int nthreads = 64;
            int iters = 16;
            int nblocks = (workspace.A->rows + nthreads * iters - 1) / (nthreads * iters);
            classifyOutlierRows<<<nblocks, nthreads, 0, workspace.streams[0]>>>(
                workspace.d_num_products,
                workspace.A->rows,
                workspace.spark_kernel_size,
                iters,
                outlier_rows,
                num_outlier_rows
            );
            CHECK_CUDA_KERNEL();

            index_t h_num_outlier_rows = 0;
            CHECK_CUDA(cudaMemcpyAsync(&h_num_outlier_rows, num_outlier_rows, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
            CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));

            sparseOutlierLauncher(
                workspace.A,
                workspace.B,
                workspace.spark_outlier_kernel_size,
                workspace.spark_kernel_size,
                workspace.d_num_products_prefixsum,
                workspace.d_num_products,
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                outlier_rows,
                h_num_outlier_rows,
                workspace.d_estik_indices,
                workspace.d_estik_values,
                workspace.d_num_outputs_row,
                workspace.d_global_mem_buffer_1,
                workspace.d_global_mem_bitmap_1,
                workspace.streams.data()
            );

        }

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.numeric_usparse_outlier += timer.getElapsedTime();
            timer.startTimer(workspace.streams[1]);
        }


        sparseKernelLauncher(
            workspace.A,
            workspace.B,
            workspace.spark_kernel_size,
            workspace.d_num_products_prefixsum,
            workspace.d_num_products,
            workspace.d_estik_indices,
            workspace.d_estik_values,
            workspace.d_num_outputs_row,
            workspace.d_b_col_idx_left,
            workspace.d_b_col_idx_right,
            config.use_hybrid_accumulator,
            workspace.streams
        );

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[1]);
            timing.numeric_usparse_main += timer.getElapsedTime();
            timer.startTimer();
        }

        workspace.syncMainToStreams(2, 4);

        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage,
            workspace.temp_storage_bytes,
            workspace.d_num_outputs_row,
            workspace.d_psum_outputs_row,
            workspace.A->rows,
            workspace.streams[0]
        ));

        size_t C_nnz = 0;
        index_t last_prefix_sum = 0;
        index_t last_item_num = 0;
        CHECK_CUDA(cudaMemcpyAsync(&last_prefix_sum, workspace.d_psum_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&last_item_num, workspace.d_num_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        C_nnz = last_prefix_sum + last_item_num;

        if (workspace.C->nnz == 0) {
            workspace.C->alloc(workspace.A->rows, workspace.B->cols, C_nnz, false, workspace.streams[0]);
        }
        workspace.C->row_offsets = workspace.d_psum_outputs_row;
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) { 
            timer.stopTimer();
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        workspace.syncStreamsToMain(3, 5);

        copyKernelLauncher<index_t, size_t>(
            workspace.C->col_ids, workspace.d_estik_indices,
            workspace.C->row_offsets, workspace.d_num_products_prefixsum,
            workspace.d_num_outputs_row,
            workspace.A->rows,
            nullptr,
            (index_t)workspace.ana1_avg_product,
            workspace.streams[1]
        );

        copyKernelLauncher<data_t, size_t>(
            workspace.C->data, workspace.d_estik_values,
            workspace.C->row_offsets, workspace.d_num_products_prefixsum,
            workspace.d_num_outputs_row,
            workspace.A->rows,
            nullptr,
            (index_t)workspace.ana1_avg_product,
            workspace.streams[0]
        );

        workspace.syncMainToStreams(3, 6);

        CHECK_CUDA_KERNEL();
        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.epilogue_copy += timer.getElapsedTime();
        }
        return;
    }


    void numeric_kernel_precise(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {

        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer();
        }

        // firstly, do a prefix scan on the d_num_outputs_row;
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage,
            workspace.temp_storage_bytes,
            workspace.d_num_outputs_row,
            workspace.d_psum_outputs_row,
            workspace.A->rows,
            workspace.streams[0]
        ));

        size_t C_nnz = 0;
        index_t last_prefix_sum = 0;
        index_t last_item_num = 0;
        CHECK_CUDA(cudaMemcpyAsync(&last_prefix_sum, workspace.d_psum_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&last_item_num, workspace.d_num_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        C_nnz = last_prefix_sum + last_item_num;

        if (workspace.C->nnz == 0) {
            workspace.C->alloc(workspace.A->rows, workspace.B->cols, C_nnz, false, workspace.streams[0]);
        }

        workspace.C->row_offsets = workspace.d_psum_outputs_row;
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) {
            timer.stopTimer();
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        // secondly, apply binning
        // here binning is tricky
        CHECK_CUDA(cudaMemsetAsync(workspace.d_bin_count, 0, sizeof(index_t) * BIN_NUM * 2, workspace.streams[0]));
        index_t* bin_count = workspace.d_bin_count;
        index_t* bins = workspace.d_bins;

        const int items_per_block = BINNING_K_ITER_PER_BLOCK * BINNING_K_BLOCK_SIZE;
        int nblocks = (workspace.A->rows + items_per_block - 1) / items_per_block;

        numericBinning<true><<<nblocks, BINNING_K_BLOCK_SIZE, 0, workspace.streams[0]>>>(
            workspace.d_num_outputs_row, 
            bin_count, 
            bins,
            workspace.d_b_col_idx_left,
            workspace.d_b_col_idx_right,
            workspace.A->rows, 
            items_per_block,
            config.numeric_expansion_coe,
            workspace.hash_numeric_use_largest,
            workspace.d_hybrid_hashmap_cnt,
            workspace.hybrid_hashmap_max
        );

        CHECK_CUDA_KERNEL();

        workspace.h_bin_count.resize(BIN_NUM * 2);
        index_t* h_bin_count = workspace.h_bin_count.data();
        CHECK_CUDA(cudaMemcpyAsync(h_bin_count, bin_count, sizeof(index_t) * (BIN_NUM * 2), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));

        if (config.print_info) {
            workspace.bin2 = workspace.h_bin_count;
        }
        // std::cout << "Should not use dense bins! " << std::endl;

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.numeric_binning += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        // allocate mem for dense kernel
        index_t alloc_size = sizeof(index_t) * workspace.B->rows * LARGEST_KERNEL_NUM_BUFFERS;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_bitmap_1, sizeof(uint32_t) * LARGEST_KERNEL_NUM_BUFFERS, workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_bitmap_1, 0, sizeof(uint32_t) * LARGEST_KERNEL_NUM_BUFFERS, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_buffer_1, alloc_size, workspace.streams[0]));


        index_t num_hash_buffers = 0;
        if (workspace.h_bin_count[BIN_NUM-1] != 0) {
            // num_hash_buffers = workspace.h_bin_count[BIN_NUM-1];
            num_hash_buffers = LARGEST_KERNEL_NUM_BUFFERS;
            index_t alloc_size = sizeof(data_t) * HASH_NUMERIC_BIN_SIZES[BIN_NUM-1] * num_hash_buffers;
            CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_buffer_3, alloc_size, workspace.streams[0]));
            CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_bitmap_3, sizeof(uint32_t) * num_hash_buffers, workspace.streams[0]));
            CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_bitmap_3, 0, sizeof(uint32_t) * num_hash_buffers, workspace.streams[0]));
        }

        workspace.syncStreamsToMain(BIN_NUM*2, 3);

        CHECK_CUDA_KERNEL();

        hashNumericLauncher<false>(
            workspace.A, workspace.B,
            workspace.h_bin_count, workspace.d_bins,
            workspace.d_num_products, 
            workspace.d_max_b_row_len,
            workspace.d_psum_outputs_row,
            workspace.C->col_ids, workspace.C->data,
            workspace.streams,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            workspace.d_global_mem_buffer_3,
            workspace.d_global_mem_bitmap_3,
            num_hash_buffers
        );

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.numeric_hash += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        if (workspace.dense_query_bitmap) {
            denseNumericLauncher<false, true>(
                workspace.A, workspace.B,
                workspace.h_bin_count.data() + BIN_NUM, workspace.d_bins + BIN_NUM * workspace.A->rows,
                workspace.d_num_products, 
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.d_psum_outputs_row,
                workspace.C->col_ids, workspace.C->data,
                workspace.streams.data() + BIN_NUM,
                workspace.d_global_mem_buffer_1,
                alloc_size,
                workspace.d_global_mem_bitmap_1
            );
        } else {
            denseNumericLauncher<false, false>(
                workspace.A, workspace.B,
                workspace.h_bin_count.data() + BIN_NUM, workspace.d_bins + BIN_NUM * workspace.A->rows,
                workspace.d_num_products, 
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.d_psum_outputs_row,
                workspace.C->col_ids, workspace.C->data,
                workspace.streams.data() + BIN_NUM,
                workspace.d_global_mem_buffer_1,
                alloc_size,
                workspace.d_global_mem_bitmap_1
            );
        }

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.numeric_dense += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        outputSortingDynLauncher(
            workspace.C->col_ids, workspace.C->data, workspace.C->row_offsets,
            workspace.C->col_ids, workspace.C->data, workspace.C->row_offsets,
            workspace.h_bin_count.data(),
            workspace.d_bins,
            workspace.d_num_outputs_row,
            workspace.streams,
            workspace.A->rows,
            workspace.B->cols,
            !config.use_indirect_sort
        );

        workspace.syncMainToStreams(BIN_NUM*2, 4);

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.epilogue_sort += timer.getElapsedTime();
        }

    }

    void numeric_kernel_est(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        // start timer
        cuTimer timer;
        if (config.track_stage_times) {
            timer.startTimer();
        }
        CHECK_CUDA(cudaMallocAsync(&workspace.d_overflow_rows, sizeof(index_t) * workspace.A->rows, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_bin_count, sizeof(index_t) * BIN_NUM * 2 * workspace.num_sections, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_bins, sizeof(index_t) * workspace.A->rows * workspace.num_sections * BIN_NUM * 2, workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_bin_count, 0, sizeof(index_t) * BIN_NUM * 2 * workspace.num_sections, workspace.streams[0]));
        if (config.track_stage_times) {
            timer.stopTimer();
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer();
        }

        // do the binning
        index_t* bin_count = workspace.d_bin_count;
        index_t* bins = workspace.d_bins;


        const int items_per_block = BINNING_K_ITER_PER_BLOCK * BINNING_K_BLOCK_SIZE;
        int nblocks = (workspace.A->rows + items_per_block - 1) / items_per_block;

        numericBinning<false, true><<<nblocks, BINNING_K_BLOCK_SIZE, 0, workspace.streams[0]>>>(
            workspace.d_est, 
            bin_count, 
            bins,
            workspace.d_b_col_idx_left,
            workspace.d_b_col_idx_right,
            workspace.A->rows, 
            items_per_block,
            0.0,
            workspace.hash_numeric_use_largest,
            workspace.d_hybrid_hashmap_cnt,
            workspace.hybrid_hashmap_max
        );

        CHECK_CUDA_KERNEL();

        // get the prefix sum of estimated C nnz
        // TODO: put into different streams.
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage, workspace.temp_storage_bytes,
            workspace.d_est, workspace.d_psum_est, workspace.A->rows * workspace.num_sections,
            workspace.streams[0]
        ));

        // copy the last element of prefix sum and also last element of est to calculate the total number
        size_t last_prefix_sum = 0;
        index_t last_item_num = 0;
        CHECK_CUDA(cudaMemcpyAsync(&last_prefix_sum, workspace.d_psum_est + workspace.A->rows * workspace.num_sections - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&last_item_num, workspace.d_est + workspace.A->rows * workspace.num_sections - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        workspace.total_est = last_prefix_sum + last_item_num;
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.numeric_binning += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        double expansion = config.hll_numeric_malloc_expansion_coe;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_estik_indices, sizeof(index_t) * workspace.total_est * expansion, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_estik_values, sizeof(data_t) * workspace.total_est * expansion, workspace.streams[0]));

        CHECK_CUDA_KERNEL();


        workspace.h_bin_count.resize(BIN_NUM * 2);
        index_t* h_bin_count = workspace.h_bin_count.data();
        CHECK_CUDA(cudaMemcpyAsync(h_bin_count, bin_count, sizeof(index_t) * (BIN_NUM * 2), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));

        if (config.print_info) {
            workspace.bin2 = workspace.h_bin_count;
        }

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        size_t alloc_size = sizeof(index_t) * workspace.B->rows * LARGEST_KERNEL_NUM_BUFFERS;
        CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_bitmap_1, sizeof(uint32_t) * LARGEST_KERNEL_NUM_BUFFERS, workspace.streams[0]));
        CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_bitmap_1, 0, sizeof(uint32_t) * LARGEST_KERNEL_NUM_BUFFERS, workspace.streams[0]));
        CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_buffer_1, alloc_size, workspace.streams[0]));

        index_t num_hash_buffers = 0;
        if (workspace.h_bin_count[BIN_NUM-1] != 0) {
            num_hash_buffers = workspace.h_bin_count[BIN_NUM-1];
            if (num_hash_buffers > LARGEST_KERNEL_NUM_BUFFERS) {
                num_hash_buffers = LARGEST_KERNEL_NUM_BUFFERS;
            }
            index_t alloc_size = sizeof(data_t) * HASH_NUMERIC_BIN_SIZES[BIN_NUM-1] * num_hash_buffers;
            CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_buffer_3, alloc_size, workspace.streams[0]));
            CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_bitmap_3, sizeof(uint32_t) * num_hash_buffers, workspace.streams[0]));
            CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_bitmap_3, 0, sizeof(uint32_t) * num_hash_buffers, workspace.streams[0]));
        }

        workspace.syncStreamsToMain(BIN_NUM*2, 3);  // reuse label 3

        hashNumericLauncher<true>(
            workspace.A, workspace.B,
            workspace.h_bin_count, workspace.d_bins,
            workspace.d_num_products, 
            workspace.d_max_b_row_len,
            workspace.d_psum_est,
            workspace.d_estik_indices, workspace.d_estik_values,
            workspace.streams,
            workspace.d_num_outputs_row,
            workspace.d_num_overflow_rows,
            workspace.d_overflow_rows,
            workspace.d_overflow_buffer_allocated,
            workspace.d_global_mem_buffer_3,
            workspace.d_global_mem_bitmap_3,
            num_hash_buffers
        );

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.numeric_hash += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        if (workspace.dense_query_bitmap) {
            denseNumericLauncher<true, true>(
                workspace.A, workspace.B,
                workspace.h_bin_count.data() + BIN_NUM, workspace.d_bins + BIN_NUM * workspace.A->rows,
                workspace.d_num_products, 
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.d_psum_est,
                workspace.d_estik_indices, workspace.d_estik_values,
                workspace.streams.data() + BIN_NUM,
                workspace.d_global_mem_buffer_1,
                alloc_size,
                workspace.d_global_mem_bitmap_1,
                workspace.d_num_outputs_row,
                workspace.d_est,
                workspace.d_num_overflow_rows,
                workspace.d_overflow_rows,
                workspace.d_overflow_buffer_allocated
            );
        } else {
            denseNumericLauncher<true, false>(
                workspace.A, workspace.B,
                workspace.h_bin_count.data() + BIN_NUM, workspace.d_bins + BIN_NUM * workspace.A->rows,
                workspace.d_num_products, 
                workspace.d_max_b_row_len,
                workspace.d_b_col_idx_left,
                workspace.d_b_col_idx_right,
                workspace.d_psum_est,
                workspace.d_estik_indices, workspace.d_estik_values,
                workspace.streams.data() + BIN_NUM,
                workspace.d_global_mem_buffer_1,
                alloc_size,
                workspace.d_global_mem_bitmap_1,
                workspace.d_num_outputs_row,
                workspace.d_est,
                workspace.d_num_overflow_rows,
                workspace.d_overflow_rows,
                workspace.d_overflow_buffer_allocated
            );
        }

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.numeric_dense += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        workspace.syncMainToStreams(BIN_NUM*2, 4);  // reuse label 4

        // copy number of overflow rows
        // here we need to do a global scan so we cannot overlap sorting with kernel anyway
        CHECK_CUDA(cudaMemcpyAsync(&workspace.h_num_overflow_rows, workspace.d_num_overflow_rows, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&workspace.h_overflow_buffer_allocated, workspace.d_overflow_buffer_allocated, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));

        if (workspace.h_num_overflow_rows > 0) {

            if (workspace.total_est * (expansion - 1) < workspace.h_overflow_buffer_allocated) {
                std::cout << "The extra allocated buffer is not enough." << std::endl;
                throw std::runtime_error("Not enough overflow buffer allocated. Try to increase hll_numeric_malloc_expansion_coe (default at 1.3).");
            } else {
                index_t buffer_size = workspace.B->rows;
                index_t num_buffers = LARGEST_KERNEL_NUM_BUFFERS;
                CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_buffer_2, sizeof(index_t) * num_buffers * buffer_size, workspace.streams[0]));
                CHECK_CUDA(cudaMallocAsync(&workspace.d_global_mem_bitmap_2, sizeof(uint32_t) * num_buffers, workspace.streams[0]));
                CHECK_CUDA(cudaMemsetAsync(workspace.d_global_mem_bitmap_2, 0x00, sizeof(uint32_t) * num_buffers, workspace.streams[0]));
                cudaFuncSetAttribute(denseNumericIterKernel<10, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHARED_MEM_PER_SM);

                denseNumericIterKernel<10, false, true><<<workspace.h_num_overflow_rows, DENSE_NUMERIC_LARGEST_BLOCK_SIZE, DENSE_NUMERIC_LARGEST_SMEM_SIZE, workspace.streams[0]>>>(
                    workspace.A->row_offsets, workspace.A->col_ids, workspace.A->data, workspace.A->rows, workspace.A->cols, workspace.A->nnz,
                    workspace.B->row_offsets, workspace.B->col_ids, workspace.B->data, workspace.B->rows, workspace.B->cols, workspace.B->nnz,
                    workspace.d_psum_est, workspace.d_estik_indices + workspace.total_est, workspace.d_estik_values + workspace.total_est,
                    workspace.d_num_products, workspace.d_max_b_row_len,
                    workspace.d_b_col_idx_left, workspace.d_b_col_idx_right,
                    workspace.d_overflow_rows,
                    DENSE_NUMERIC_LARGEST_ITER_SIZE,
                    // workspace.h_num_overflow_rows,
                    DENSE_NUMERIC_LARGEST_ROW_OFFSET_SMEM,
                    workspace.d_global_mem_buffer_2,
                    workspace.d_global_mem_bitmap_2,
                    workspace.d_num_outputs_row
                );
                CHECK_CUDA_KERNEL();
            }
        } else {
            // do nothing
        }

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.numeric_overflow += timer.getElapsedTime();
            timer.startTimer(workspace.streams[0]);
        }

        // the first step: prefix sum output sections        
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            workspace.d_temp_storage, workspace.temp_storage_bytes,
            workspace.d_num_outputs_row, workspace.d_psum_outputs_row, workspace.A->rows,
            workspace.streams[0]
        ));

        size_t C_nnz = 0;
        index_t last_prefix_sum_2 = 0;
        index_t last_item_num_2 = 0;
        CHECK_CUDA(cudaMemcpyAsync(&last_prefix_sum_2, workspace.d_psum_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaMemcpyAsync(&last_item_num_2, workspace.d_num_outputs_row + workspace.A->rows - 1, sizeof(index_t), cudaMemcpyDeviceToHost, workspace.streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(workspace.streams[0]));
        C_nnz = last_prefix_sum_2 + last_item_num_2;

        if (workspace.C->nnz == 0) {
            workspace.C->alloc(workspace.A->rows, workspace.B->cols, C_nnz, false, workspace.streams[0]);
        }
        workspace.C->row_offsets = workspace.d_psum_outputs_row;
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.numeric_malloc += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        index_t avg_elements_per_row = C_nnz / workspace.A->rows;

        // step 2: do the copies (without sorting)
        // that would count the first 3 groups of hash kernel and all dense kernels
        // do the copies for the overflow bin

        workspace.syncStreamsToMain(BIN_NUM*2+2, 5);

        if (workspace.h_num_overflow_rows > 0) {
            copyKernelLauncher<index_t>(
                workspace.C->col_ids, workspace.d_estik_indices + workspace.total_est,
                workspace.C->row_offsets, workspace.d_psum_est,
                workspace.d_num_outputs_row, workspace.h_num_overflow_rows,
                workspace.d_overflow_rows,
                avg_elements_per_row * 2,   // rough est
                workspace.streams[0]
            );

            copyKernelLauncher<data_t>(
                workspace.C->data, workspace.d_estik_values + workspace.total_est,
                workspace.C->row_offsets, workspace.d_psum_est,
                workspace.d_num_outputs_row, workspace.h_num_overflow_rows,
                workspace.d_overflow_rows,
                avg_elements_per_row * 2,   // rough est
                workspace.streams[0]
            );
        }

        CHECK_CUDA_KERNEL();

        // we are not using multiple group of streams, as outputCopyingLauncher is using only first two streams
        outputCopyingLauncher(
            workspace.d_estik_indices, workspace.d_estik_values,
            workspace.d_psum_est,
            workspace.C->col_ids, workspace.C->data,
            workspace.C->row_offsets,
            workspace.h_bin_count.data() + BIN_NUM,
            workspace.d_bins + BIN_NUM * workspace.A->rows,
            workspace.d_num_outputs_row,
            workspace.streams,
            workspace.A->rows,
            avg_elements_per_row
        );

        CHECK_CUDA_KERNEL();

        outputCopyingLauncher(
            workspace.d_estik_indices, workspace.d_estik_values,
            workspace.d_psum_est,
            workspace.C->col_ids, workspace.C->data,
            workspace.C->row_offsets,
            workspace.h_bin_count.data(),
            workspace.d_bins,
            workspace.d_num_outputs_row,
            workspace.streams,
            workspace.A->rows,
            avg_elements_per_row,
            2
        );
        CHECK_CUDA_KERNEL();

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams);
            timing.epilogue_copy += timer.getElapsedTime();
            timer.startTimer(workspace.streams);
        }

        outputSortingDynLauncher(
            workspace.d_estik_indices, workspace.d_estik_values, workspace.d_psum_est,
            workspace.C->col_ids, workspace.C->data, workspace.C->row_offsets,
            workspace.h_bin_count.data(),
            workspace.d_bins,
            workspace.d_num_outputs_row,
            workspace.streams,
            workspace.A->rows,
            workspace.B->cols,
            !config.use_indirect_sort
        );

        CHECK_CUDA_KERNEL();

        workspace.syncMainToStreams(BIN_NUM*2+2, 6);

        if (config.track_stage_times) {
            timer.stopTimer(workspace.streams[0]);
            timing.epilogue_sort += timer.getElapsedTime();
        }

    }

    /* Free the temp vars */
    /* A, B, and C are not deallocated */
    void deallocate(
        Workspace& workspace,
        Config& config,
        Timing& timing
    ) {
        cudaDeviceSynchronize();
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_num_products, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_num_products_prefixsum, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_max_b_row_len, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_avg_b_row_len, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_b_col_idx_left, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_b_col_idx_right, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_max_product, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_max_a_row_nnz, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_total_sample_rows_nnz, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_total_sampled_products, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_sampled_compaction_sum, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_num_early_exit_rows, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_sampled_compaction, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_num_overflow_rows, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_overflow_buffer_allocated, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_outlier_rows, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_num_outlier_rows, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_hll_b, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_hll_c, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_est, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_bin_count, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_bins, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_psum_est, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_temp_storage, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_buffer_0, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_bitmap_0, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_buffer_1, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_bitmap_1, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_buffer_2, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_bitmap_2, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_buffer_3, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_global_mem_bitmap_3, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_estik_indices, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_estik_values, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_overflow_rows, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_num_outputs_row, workspace.streams[0]);
        CHECK_AND_FREE_DEVICE_MEM_ASYNC(workspace.d_hybrid_hashmap_cnt, workspace.streams[0]);
        cudaDeviceSynchronize();

    }


    json save_workspace_to_json(Workspace& workspace, Config& config) {
        // information to log:
        
        // Part A: input matrix dimensions
        json j;
        j["A"] = { {"rows", workspace.A->rows}, {"cols", workspace.A->cols}, {"nnz", workspace.A->nnz} };
        j["B"] = { {"rows", workspace.B->rows}, {"cols", workspace.B->cols}, {"nnz", workspace.B->nnz} };
        
        // Part B: output matrix dimensions
        j["C"] = { {"rows", workspace.C->rows}, {"cols", workspace.C->cols}, {"nnz", workspace.C->nnz} };

        // Part C: The intermediate product information
        j["stats"]["intermediate_product"] = workspace.total_products;
        j["stats"]["intermediate_product_max"] = workspace.h_max_product;
        j["stats"]["intermediate_product_avg"] = ((double)workspace.total_products) / workspace.A->rows;
        j["stats"]["input_expansion"] = ((double)workspace.total_products) / workspace.A->nnz;
        j["stats"]["output_compression"] = ((double)workspace.total_products) / (workspace.C->nnz);

        // Matrix workflow selection
        int workflow = 0;
        if (workspace.ana1_type == 0) {
            workflow = 2;
        } else if (workspace.ana1_type == 2) {
            workflow = 3;
        } else if (workspace.ana2_type == 0) {
            workflow = 0;
        } else {
            workflow = 1;
        }
        j["decision"]["workflow"] = workflow;

        // sampling
        if (workflow <= 1) {
            j["sampling"]["compaction_1"] = workspace.ana2_avg_sampled_compaction;
            j["sampling"]["compaction_2"] = workspace.ana2_avg_sampled_compaction2;
            j["binning_2"] = workspace.bin2;
        }
        if (workflow <= 0) {
            j["sampling"]["compaction_2_var"] = workspace.compaction_var;
            j["sampling"]["safe_estimated_compaction"] = workspace.safe_estimated_avg_compaction;
            j["decision"]["estimated_symbolic"] = workspace.use_estimated_symbolic;
            j["binning_1"] = workspace.bin1;
        }

        // Decisions
        if (workflow <= 1) {
            j["decision"]["hash_largest"] = workspace.hash_numeric_use_largest;
            j["decision"]["dense_bitmap_query"] = workspace.dense_query_bitmap;
        }

        if (workflow == 1) {
            j["estimation"]["overflow_rows"] = workspace.h_num_overflow_rows;
            j["estimation"]["log2_precision"] = workspace.hll_log2_precision;
            j["estimation"]["total_est"] = workspace.total_est;
            j["estimation"]["expansion"] = config.hll_numeric_malloc_expansion_coe;
        }


        // Sparse kernel information (workflow 2)
        if (workflow == 2) {
            j["ultra_sparse"] = {{"kernel_size", workspace.spark_kernel_size}, {"outlier_size", workspace.spark_outlier_kernel_size}};
        }

        // Segmentation (spy-plot image segmentation) results
        if (workspace.segmentation_ran) {
            const auto& s = workspace.segmentation;
            j["segmentation"]["grid_h"]          = s.grid_h;
            j["segmentation"]["grid_w"]          = s.grid_w;
            j["segmentation"]["tile_rows"]       = s.tile_rows;
            j["segmentation"]["tile_cols"]       = s.tile_cols;
            j["segmentation"]["density_mean"]    = s.density_mean;
            j["segmentation"]["density_stddev"]  = s.density_stddev;
            j["segmentation"]["num_components"]  = s.num_components;
            j["segmentation"]["cc_iterations"]   = s.cc_iterations;
            j["segmentation"]["time_density_ms"]   = s.time_density_ms;
            j["segmentation"]["time_threshold_ms"] = s.time_threshold_ms;
            j["segmentation"]["time_cc_ms"]        = s.time_cc_ms;
            j["segmentation"]["time_classify_ms"]  = s.time_classify_ms;

            json segs = json::array();
            for (const auto& seg : s.segments) {
                json sj;
                sj["row_start"]     = seg.row_start;
                sj["row_end"]       = seg.row_end;
                sj["row_count"]     = seg.row_count;
                sj["col_left"]      = seg.col_left;
                sj["col_right"]     = seg.col_right;
                sj["class"]         = segmentClassName(seg.cls);
                sj["sum_products"]  = seg.sum_products;
                sj["sum_a_nnz"]     = seg.sum_a_nnz;
                sj["mean_row_len"]  = seg.mean_row_len;
                sj["var_row_len"]   = seg.var_row_len;
                sj["local_er"]      = seg.local_er;
                sj["max_b_row_len"] = seg.max_b_row_len;
                sj["max_a_row_len"] = seg.max_a_row_len;
                segs.push_back(sj);
            }
            j["segmentation"]["segments"] = segs;
        }

        return j;
        // should we log the config struct?
    }

};


}
