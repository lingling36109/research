#pragma once
#include "Common.h"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace somespgemm{

struct CSR;
struct cuCSR;

void convert(cuCSR& dst, const CSR& src, unsigned int padding = 0);
void convert(CSR& dst, const cuCSR& scr, unsigned int padding = 0);

struct CSR
{
	void computeStatistics(double& mean, double& std_dev, size_t& max, size_t& min);

	size_t rows, cols, nnz;

	std::unique_ptr<data_t[]> data;
	std::unique_ptr<index_t[]> row_offsets;
	std::unique_ptr<index_t[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0) { }
	CSR(cuCSR& src) : rows(0), cols(0), nnz(0), data(nullptr), row_offsets(nullptr), col_ids(nullptr) {
		convert(*this, src);
	}
	void alloc(size_t rows, size_t cols, size_t nnz);
};

std::shared_ptr<CSR> loadCSR(const char* file);

struct cuCSR {
	size_t rows, cols, nnz;

	data_t* data;
	index_t* row_offsets;
	index_t* col_ids;

	cuCSR() : rows(0), cols(0), nnz(0), data(nullptr), row_offsets(nullptr), col_ids(nullptr) { }
	cuCSR(const CSR& src) : rows(0), cols(0), nnz(0), data(nullptr), row_offsets(nullptr), col_ids(nullptr) {
		convert(*this, src);
	}
	void alloc(size_t rows, size_t cols, size_t nnz, bool allocOffsets = true);
	void alloc(size_t r, size_t c, size_t n, bool allocOffsets, cudaStream_t stream);
	void reset();
	void reset(cudaStream_t stream);
	virtual ~cuCSR();
};

void compare(const CSR& a, const CSR& b);

}


