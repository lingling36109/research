#pragma once

#include "CSR.h"
#include "Common.h"
#include "Json.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <map>

using json = nlohmann::json;

namespace somespgemm {

struct Config {

    /* These are algorithm-related settings */

    // Switches and thresholds
    bool use_estimation_workflow = true;
    bool use_ultrasparse_workflow = true;
    bool use_fat_workflow = true;
    bool use_assist_information = true;
    bool use_hybrid_accumulator = true;
    bool use_indirect_sort = true;

    // Segmentation (spy-plot image segmentation) settings
    bool use_segmented_workflow = false;
    int segmentation_grid_max = 256;
    double segmentation_hot_threshold_k = 2.0;
    int segmentation_min_segment_rows = 1024;
    int segmentation_cc_max_iters = 32;

    int usparse_avg_product_threshold = 64;
    int estimation_input_expansion_threshold = 8;
    int estimation_output_compaction_threshold = 8;
    int dense_query_bitmap_compaction_threshold = 2;
    // the hybrid numeric hash kernel would be enabled,
    // only if output row length exceeds the threshold
    // TODO: change to hardware-related coefficiency
    int numeric_hybrid_hash_kernel_row_length_threshold = 70000;
    int fat_ratio = 100;

    // HLL Estimator 
    double hll_sample_rate = 0.03;
    int hll_sample_min_rows = 648;
    int hll_sample_max_rows = 10800;

    // empirical expansion factors for hll estimated results
    // only used at estimation-based numeric kernel
    const std::map<int, double> hll_expansion_coes = {
        {5, 2.0},
        {6, 1.5}
    };
    double hll_numeric_malloc_expansion_coe = 1.3;
    int hll_higher_precision_expansion_threshold = 48;
    int hll_precision_low = 5;
    int hll_precision_high = 6;


    // For estimation-assisted Symbolic Kernel
    const std::map<double, double> normal_distribution_z_values = {
        {0.80, 1.28},
        {0.85, 1.44},
        {0.90, 1.64},
        {0.95, 1.96},
        {0.99, 2.53},
        {0.999, 3.30}
    };

    double estimate_compaction_target_confidence_interval = 0.99;
    double estimate_compaction_target_z_value = 2.53;

    void set_estimate_compaction(double confidence_interval) {
        if (normal_distribution_z_values.find(confidence_interval) != normal_distribution_z_values.end()) {
            estimate_compaction_target_confidence_interval = confidence_interval;
            estimate_compaction_target_z_value = normal_distribution_z_values.at(confidence_interval);
        } else {
            std::cerr << "Warning: Confidence interval not found. Using default z-value." << std::endl;
            throw std::invalid_argument("Invalid confidence interval");
        }
    }

    // only used at non-estimation-based kernels
    double symbolic_expansion_coe = 1.25;
    double numeric_expansion_coe = 1.5; 

    // double symbolic_hash_gmem_init_coe = 1.5;
    // double symbolic_hash_gmem_map_expansion_coe = 1.25;

    /* These are non-algorithm-related settings */
    bool track_stage_times = false;

    bool check_correctness = false;

    bool output_stats = false;
    double compare_relative_error = 1e-6;
    double compare_autofail_relative_error = 1e-4;
    std::string ground_truth_file = "";
    std::string stats_output_file = "";

    int warmup_iters = 10, bench_iters = 10;

    bool print_info = false;
    bool print_memory = false;
};

template <typename T>
void assign_if_present(const json& j, const char* key, T& field) {
    if (j.contains(key))
        j.at(key).get_to(field);
}

void from_json(const json& j, Config& c);

std::shared_ptr<CSR> loadMatrix(std::string path);

class cuTimer {
public:
    cuTimer() : start(0), stop(0) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void startTimer() { cudaEventRecord(start, 0); }
    void startTimer(cudaStream_t stream) { cudaEventRecord(start, stream); }
    void startTimer(std::vector<cudaStream_t>& streams) {
        // just force synchronize
        cudaDeviceSynchronize();
        cudaEventRecord(start, streams[0]);
    }

    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    }

    void stopTimer(cudaStream_t stream) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
    }

    void stopTimer(std::vector<cudaStream_t>& streams) {
        // just force synchronize
        cudaDeviceSynchronize();
        cudaEventRecord(stop, streams[0]);
        cudaEventSynchronize(stop);
    }


    float getElapsedTime() {
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return elapsedTime;
    }
    
    void reportElapsedTime(std::string msg) {
        float elapsedTime = getElapsedTime();
        std::cout << msg << " elapsed time: " << elapsedTime << " ms\n";
    }

    ~cuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

private:
    cudaEvent_t start;
    cudaEvent_t stop;
};


struct Timing {
    double prologue;

    double analysis_product_calc;
    double analysis_reduce;
    double analysis_mem_cpy;

    double sampling_var_calc_host;

    double est_sample;
    double est_malloc;
    double est_hll_construct;
    double est_hll_merge;
    double est_binning;

    double symbolic_binning;
    double symbolic_hash;
    double symbolic_dense;

    double numeric_malloc;
    double numeric_binning;
    double numeric_usparse_outlier;
    double numeric_usparse_main;
    double numeric_fat;
    double numeric_hash;
    double numeric_dense;
    double numeric_overflow;

    double epilogue_scan;
    double epilogue_sort;
    double epilogue_copy;

    Timing() : prologue(0), analysis_product_calc(0), analysis_reduce(0), analysis_mem_cpy(0), 
        sampling_var_calc_host(0),
        est_sample(0), est_hll_construct(0), est_hll_merge(0), est_binning(0), est_malloc(0),
        symbolic_binning(0), symbolic_hash(0), symbolic_dense(0),
        numeric_malloc(0), numeric_binning(0), numeric_usparse_outlier(0), numeric_usparse_main(0),
        numeric_fat(0),
        numeric_hash(0), numeric_dense(0), numeric_overflow(0),
        epilogue_scan(0), epilogue_sort(0), epilogue_copy(0)
        { }

    void reset() {
        prologue = 0;
        analysis_product_calc = 0;
        analysis_reduce = 0;
        analysis_mem_cpy = 0;
        sampling_var_calc_host = 0;
        est_sample = 0;
        est_hll_construct = 0;
        est_hll_merge = 0;
        est_binning = 0;
        est_malloc = 0;
        symbolic_binning = 0;
        symbolic_hash = 0;
        symbolic_dense = 0;
        numeric_malloc = 0;
        numeric_binning = 0;
        numeric_usparse_outlier = 0;
        numeric_usparse_main = 0;
        numeric_fat = 0;
        numeric_hash = 0;
        numeric_dense = 0;
        numeric_overflow = 0;
        epilogue_scan = 0;
        epilogue_sort = 0;
        epilogue_copy = 0;
    }

    void report(int iters, size_t total_flops=0, double total_time=0.0) {
        std::cout << BANNER << "\n";
        if (total_flops > 0 && total_time > 0.0) {
            double gflops = (double)total_flops / 1e9;
            double seconds = total_time / 1e3;
            std::cout << "Overall GFLOPS: " << gflops / seconds << " GFLOPS\n";
            if (symbolic_dense + symbolic_hash + symbolic_binning > 0) {
                double symbolic_time = symbolic_dense + symbolic_hash + symbolic_binning;
                double symbolic_seconds = symbolic_time / iters / 1e3;
                std::cout << "Symbolic GFLOPS: " << gflops / symbolic_seconds << " GFLOPS\n";
            }
            if (numeric_dense + numeric_hash + numeric_usparse_main + numeric_usparse_outlier + numeric_binning > 0) {
                double numeric_time = numeric_dense + numeric_hash + numeric_usparse_main + numeric_usparse_outlier + numeric_binning;
                double numeric_seconds = numeric_time / iters / 1e3;
                std::cout << "Numeric GFLOPS: " << gflops / numeric_seconds << " GFLOPS\n";
            }
        }

        std::cout << "Timing Report (ms):\n";
        std::cout << "Prologue: " << prologue / iters << "\n";
        std::cout << "Analysis - Product Calculation: " << analysis_product_calc / iters << "\n";
        std::cout << "Analysis - Reduction: " << analysis_reduce / iters << "\n";
        std::cout << "Analysis - Memory Copy: " << analysis_mem_cpy / iters << "\n";
        std::cout << "Estimation - Sampling: " << est_sample / iters << "\n";
        std::cout << "Estimation - Malloc: " << est_malloc / iters << "\n";
        std::cout << "Estimation - HLL Construct: " << est_hll_construct / iters << "\n";
        std::cout << "Sampling - Variance Calculation (host): " << sampling_var_calc_host / iters << "\n";
        std::cout << "Estimation - HLL Merge: " << est_hll_merge / iters << "\n";
        std::cout << "Estimation - Binning: " << est_binning / iters << "\n";
        std::cout << "Symbolic - Binning: " << symbolic_binning / iters << "\n";
        std::cout << "Symbolic - Hash: " << symbolic_hash / iters << "\n";
        std::cout << "Symbolic - Dense: " << symbolic_dense / iters << "\n";
        std::cout << "Numeric - Malloc: " << numeric_malloc / iters << "\n";
        std::cout << "Numeric - Binning: " << numeric_binning / iters << "\n";
        std::cout << "Numeric - Hash: " << numeric_hash / iters << "\n";
        std::cout << "Numeric - USparse Outlier: " << numeric_usparse_outlier / iters << "\n";
        std::cout << "Numeric - USparse Main: " << numeric_usparse_main / iters << "\n";
        std::cout << "Numeric - Dense: " << numeric_dense / iters << "\n";
        std::cout << "Numeric - Overflow Handling: " << numeric_overflow / iters << "\n";
        std::cout << "Numeric - FAT: " << numeric_fat / iters << "\n";
        std::cout << "Epilogue - Scan: " << epilogue_scan / iters << "\n";
        std::cout << "Epilogue - Sort: " << epilogue_sort / iters << "\n";
        std::cout << "Epilogue - Copy: " << epilogue_copy / iters << "\n";
        // std::cout << "Epilogue - Copy Value: " << epilogue_copy_val / iters << "\n";
        std::cout << BANNER << "\n\n\n";
    }

    json save_to_json(int iters) {
        json j;
        j["prologue"] = prologue / iters;
        j["analysis"]["product_calc"] = analysis_product_calc / iters;
        j["analysis"]["reduce"] = analysis_reduce / iters;
        j["analysis"]["mem_cpy"] = analysis_mem_cpy / iters;
        j["sampling"]["var_calc_host"] = sampling_var_calc_host / iters;
        j["estimation"]["sampling"] = est_sample / iters;
        j["estimation"]["malloc"] = est_malloc / iters;
        j["estimation"]["hll_construct"] = est_hll_construct / iters;
        j["estimation"]["hll_merge"] = est_hll_merge / iters;
        j["estimation"]["binning"] = est_binning / iters;
        j["symbolic"]["binning"] = symbolic_binning / iters;
        j["symbolic"]["hash"] = symbolic_hash / iters;
        j["symbolic"]["dense"] = symbolic_dense / iters;
        j["numeric"]["malloc"] = numeric_malloc / iters;
        j["numeric"]["binning"] = numeric_binning / iters;
        j["numeric"]["hash"] = numeric_hash / iters;
        j["numeric"]["usparse_outlier"] = numeric_usparse_outlier / iters;
        j["numeric"]["usparse_main"] = numeric_usparse_main / iters;
        j["numeric"]["dense"] = numeric_dense / iters;
        j["numeric"]["overflow"] = numeric_overflow / iters;
        j["numeric"]["fat"] = numeric_fat / iters;
        j["epilogue"]["scan"] = epilogue_scan / iters;
        j["epilogue"]["sort"] = epilogue_sort / iters;
        j["epilogue"]["copy"] = epilogue_copy / iters;
        return j;
    }

};

void printMemInfo();


}

