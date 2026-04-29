#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "SpGEMM.cuh"
#include "Utils.h"
#include "Json.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file> [matrix_b_file] [config_file]" << std::endl;
        std::cerr << "Matrices should end with .mtxd_.hicsr, and config should end with .json." << std::endl;
        return 1;
    }

    std::string matrixFile = argv[1];

    bool has_config = false;
    bool has_second_matrix = false;
    std::string configFile;
    std::string secondMatrixFile = argv[1];

    if (argc == 3) {
        // check if we have second matrix or config
        std::string arg2(argv[2]);
        if (arg2.length() >= 12 && arg2.substr(arg2.length() - 12) == ".mtxd_.hicsr") {
            has_second_matrix = true;
            secondMatrixFile = argv[2];
        } else if (arg2.length() >= 5 && arg2.substr(arg2.length() - 5) == ".json") {
            has_config = true;
            configFile = argv[2];
        } else {
            has_second_matrix = true;
            secondMatrixFile = argv[2];
        }
    } else if (argc == 4) {
        has_second_matrix = true;
        secondMatrixFile = argv[2];
        has_config = true;
        configFile = argv[3];
    };
    
    auto matrix_a = somespgemm::loadMatrix(matrixFile);
    auto matrix_b = somespgemm::loadMatrix(secondMatrixFile);

    if (matrix_a->cols != matrix_b->rows) {
        std::cerr << "Incompatible matrix dimensions!" << std::endl;
        return 1;
    }

    auto config = somespgemm::Config();
    if (has_config) {
        json j;
        std::ifstream config_ifs(configFile);
        config_ifs >> j;
        from_json(j, config);
    }

    somespgemm::SpGEMM spgemm{config};

    auto d_A = std::make_shared<somespgemm::cuCSR>(*matrix_a);
    auto d_B = std::make_shared<somespgemm::cuCSR>(*matrix_b);

    auto result = spgemm.run(d_A, d_B);
    auto result_host = std::make_shared<somespgemm::CSR>(*result);

    if (config.check_correctness) {
        auto ground_truth = somespgemm::loadMatrix(config.ground_truth_file);
        somespgemm::compare(*result_host, *ground_truth);
    }

    if (config.output_stats) {
        std::cout << "Writing to " << config.stats_output_file << std::endl;
        json stats = spgemm.get_stats();
        std::ofstream stats_ofs(config.stats_output_file);
        stats_ofs << stats.dump(4);
        stats_ofs.close();
    }

    return 0;

}