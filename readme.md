# Ocean SpGEMM

This is the implementation of Ocean, a fast estimation-based SpGEMM solution on GPUs.

## Requirement

- Linux Operating System
- NVIDIA GPUs with compute capability 8.0 or higher
- CUDA Toolkit Version 12.4 or later

This repo has only been tested on A100 and H100. 
Since it has not been tested on other hardware, it is very likely to run into compile, configuration, or runtime errors. 
We are working on testing it on more hardware.

## Setup and Build

Before building, you need to setup two parameters in `include/Common.h`.
`NUM_SM` at line 14 should be set to the number of streaming multiprocessors (SMs) on your GPU. 
- For A100, it should be 108; 
- For H100, it should be 114 (PCIe) or 132 (SXM).

`SHARED_MEMORY_KB` at line 15 should be set to the amount of dynamic shared memory available per block on your GPU. 
- For hardware that has more than 128KB, including A100 and H100, this should be set to 128.
- For other GPUs, it should be selected between [80, 96, 128]. Please refer to the GPU documentation to find the largest appropriate value.

You should also setup the `Compute_Capability = sm_xx` on line 3 `Makefile`. Note that the `-gencode=arch=compute_61` on line 6 should **NOT** be changed.

Use `make -j8` in the project directory to build the binaries.
This would generate two binaries : `spgemm` and `convert`.


## Running

### Downloading Matrices

The source matrices can be downloaded from [SuiteSparse](https://sparse.tamu.edu). Please select the `mtx` format when downloading.


Matrix lists of the two dataset used for evaluation are provided in `utils/square.csv` and `utils/rectangular.csv`.
You can use the provided `download.py` script to automate the downloading process of a matrix list. Run `python download.py --filename $PATH_TO_CSV --download_dir $PATH_TO_DOWNLOAD_DIR`.

The `convert` binary can convert `.mtx` format to `.csr` format, which is the required format to run Ocean. Simply run `./convert $PATH_TO_INPUT_MTX $PATH_TO_OUTPUT_CSR`.

Here's an example of downloading and converting a single matrix `144` from suitesparse on linux. Assuming you have already setup Ocean, built the binaries, and is currently in the project directory:

```bash
# make sure you have correctly configured Ocean and built!
wget http://sparse-files.engr.tamu.edu/MM/DIMACS10/144.tar.gz
tar -xzf 144.tar.gz
./convert 144/144.mtx 144.csr
```

### Running

After building, to SpGEMM run on a csr matrix, execute

`./spgemm $PATH_TO_MAT_A [$PATH_TO_MAT_B] [CONFIG_FILE]`

The second and third parameters are optional. 
If `MAT_B` is not provided, `MAT_A` would be used as both `MAT_A` and `MAT_B`.
The config file is also optional. If not provided, the default config will be used. Please refer to Configuration section for more details on runtime configuration.

## Configuration

There are many settings that can be configured before running. 
Multiple examples of config files can be found in `config/` directory:
- `bench.json` should be used for performance benchmarking.
- `bench_detail.json` will output collected metrics and detailed timing after running the benchmark. It would incur extra synchronization cost.
- `analysis.json` shows all available setting fields.

Here are explanations for some of the setting fields:
- To compare output with another ground truth matrix (also in `csr` format), set `check_correctness` to `true`, and `ground_truth_file` to the ground truth matrix. It is recommended to use spECK's output as the ground truth matrix. Note that for certain matrices, the numerical comparison *may fail*. It is likely due to floating point error and does not indicate actual failure.
- The options starting with `use_` mostly control the optimization strategies of Ocean.
- Setting `output_stats` to `true` and `stats_output_file` to a valid file path will enable the output of runtime statistics.


## Program Structure

```
.
├── bin
├── include												# C++ header files
│   ├── Json.hpp
│   ├── Common.h
│   ├── CSR.h
│   └── Utils.h
├── kernels												# CUDA Kernels and SpGEMM main workflow file
│   ├── AccumulatorCommon.cuh
│   ├── AccumulatorDense.cuh
│   ├── AccumulatorESC.cuh
│   ├── AccumulatorHash.cuh
│   ├── Analysis.cuh
│   ├── DeviceCommon.cuh
│   ├── Epilogue.cuh
│   ├── Hashmap.cuh
│   ├── HLL.cuh
│   ├── MurmurHash.cuh
│   ├── SpGEMM.cuh
│   └── Wrappers.cuh
├── Makefile
├──  src												# C++ Source Files and main file.
│   ├── Analysis.cpp
│   ├── CSR.cpp
│   ├── main.cu
│   └── Utils.cpp
└──  utils												# Utilities (for convert format) 
    └── convert.cpp


```


## Acknowledgements

Part of this code is adpated from [spECK](https://github.com/GPUPeople/spECK).
We use [nlohmann/json](https://github.com/nlohmann/json) for JSON parsing and serialization.
We adapted mumurhash3 C implementation from PeterScott's [repo](https://github.com/PeterScott/murmur3/tree/master).
