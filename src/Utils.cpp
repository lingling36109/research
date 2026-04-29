#include "Utils.h"
#include "Common.h"
#include<string>
#include<iostream>

namespace somespgemm{

void from_json(const json& j, Config& c) {
    assign_if_present(j, "track_stage_time", c.track_stage_times);
    assign_if_present(j, "check_correctness", c.check_correctness);
    assign_if_present(j, "output_stats", c.output_stats);
    assign_if_present(j, "compare_relative_error", c.compare_relative_error);
    assign_if_present(j, "compare_autofail_relative_error", c.compare_autofail_relative_error);
    assign_if_present(j, "ground_truth_file", c.ground_truth_file);
    assign_if_present(j, "stats_output_file", c.stats_output_file);
    assign_if_present(j, "warmup_iters", c.warmup_iters);
    assign_if_present(j, "bench_iters", c.bench_iters);
    assign_if_present(j, "print_info", c.print_info);
    assign_if_present(j, "print_memory", c.print_memory);

    assign_if_present(j, "use_estimation_workflow", c.use_estimation_workflow);
    assign_if_present(j, "use_ultrasparse_workflow", c.use_ultrasparse_workflow);
    assign_if_present(j, "use_assist_information", c.use_assist_information);
    assign_if_present(j, "use_hybrid_accumulator", c.use_hybrid_accumulator);
    assign_if_present(j, "use_indirect_sort", c.use_indirect_sort);
    assign_if_present(j, "use_fat_workflow", c.use_fat_workflow);

    assign_if_present(j, "hll_numeric_malloc_expansion_coe", c.hll_numeric_malloc_expansion_coe);
    assign_if_present(j, "estimation_input_expansion_threshold", c.estimation_input_expansion_threshold);
    assign_if_present(j, "estimation_output_compaction_threshold", c.estimation_output_compaction_threshold);

    assign_if_present(j, "use_segmented_workflow", c.use_segmented_workflow);
    assign_if_present(j, "segmentation_grid_max", c.segmentation_grid_max);
    assign_if_present(j, "segmentation_hot_threshold_k", c.segmentation_hot_threshold_k);
    assign_if_present(j, "segmentation_min_segment_rows", c.segmentation_min_segment_rows);
    assign_if_present(j, "segmentation_cc_max_iters", c.segmentation_cc_max_iters);
}

std::shared_ptr<CSR> loadMatrix(std::string path) {
	std::string csrPath = path;

    std::shared_ptr<CSR> matrix = nullptr;

	try
	{
		std::cout << "trying to load csr file \"" << csrPath << "\"\n";
		matrix = loadCSR(csrPath.c_str());
		std::cout << "successfully loaded: \"" << csrPath << "\"\n";
	}
	catch (std::exception& ex)
	{
		std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
        exit(1);
	}

	return matrix;
}

void printMemInfo() {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status) {
        std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << "\n";
        return;
    }
    size_t used_byte = total_byte - free_byte;
    std::cout << BANNER << "\n";
    std::cout << "GPU memory usage: used = " << used_byte / 1024.0 / 1024.0 << " MB, free = " << free_byte / 1024.0 / 1024.0 << " MB, total = " << total_byte / 1024.0 / 1024.0 << " MB\n";
    std::cout << BANNER << "\n";
}

}

