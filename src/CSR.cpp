// Part of the code is taken from spECK's library
#include "CSR.h"

#include <algorithm>
#include <stdint.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iterator>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

namespace somespgemm {

void CSR::computeStatistics(double& mean, double& std_dev, size_t& max, size_t& min)
{
	// running variance by Welford
	size_t count = 0;
	mean = 0;
	double M2 = 0;
	max = 0;
	min = cols;
	for (size_t i = 0; i < rows; ++i)
	{
		size_t r_length = row_offsets[i + 1] - row_offsets[i];
		min = std::min(min, r_length);
		max = std::max(max, r_length);
		++count;
		double newValue = static_cast<double>(r_length);
		double delta = newValue - mean;
		mean = mean + delta / count;
		double delta2 = newValue - mean;
		M2 = M2 + delta * delta2;
	}
	if (count < 2)
		std_dev = 0;
	else
		std_dev = sqrt(M2 / (count - 1));
}

void CSR::alloc(size_t r, size_t c, size_t n)
{
	rows = r;
	cols = c;
	nnz = n;

	data = std::make_unique<data_t[]>(n);
	col_ids = std::make_unique<index_t[]>(n);
	row_offsets = std::make_unique<index_t[]>(r+1);
	// there is no guarantee that the last element is correct. Use nnz instead.
}

namespace {
	template<typename VALUE_TYPE>
	struct State
	{
		typedef VALUE_TYPE ValueType;

		ValueType scaling;
		bool transpose;

		State() : scaling(1), transpose(false) { }
		State(ValueType scaling, bool transpose) : scaling(scaling), transpose(transpose) { }
	};

	struct CSRIOHeader
	{
		static constexpr char Magic[] = { 'H','i', 1, 'C','o','m','p','s','d' };

		char magic[sizeof(Magic)];
		uint64_t typesize;
		uint64_t compresseddir;
		uint64_t indexsize;
		uint64_t fixedoffset;
		uint64_t offsetsize;
		uint64_t num_rows, num_columns;
		uint64_t num_non_zeroes;

		CSRIOHeader() = default;


		template<typename T>
		static uint64_t typeSize()
		{
			return sizeof(T);
		}

		template<typename T>
		CSRIOHeader(const CSR& mat)
		{
			for (size_t i = 0; i < sizeof(Magic); ++i)
				magic[i] = Magic[i];
			typesize = typeSize<T>();
			compresseddir = 0;
			indexsize = typeSize<uint32_t>();
			fixedoffset = 0;
			offsetsize = typeSize<uint32_t>();

			num_rows = mat.rows;
			num_columns = mat.cols;
			num_non_zeroes = mat.nnz;
		}

		bool checkMagic() const
		{
			for (size_t i = 0; i < sizeof(Magic); ++i)
				if (magic[i] != Magic[i])
					return false;
			return true;
		}
	};
	constexpr char CSRIOHeader::Magic[];
}

std::shared_ptr<CSR> loadCSR(const char * file)
{
	std::ifstream fstream(file, std::fstream::binary);
	if (!fstream.is_open())
		throw std::runtime_error(std::string("could not open \"") + file + "\"");

	CSRIOHeader header;
	State<data_t> state;
	fstream.read(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));
	if (!fstream.good())
		throw std::runtime_error("Could not read CSR header");
	if (!header.checkMagic())
		throw std::runtime_error("File does not appear to be a CSR Matrix");

	fstream.read(reinterpret_cast<char*>(&state), sizeof(state));
	if (!fstream.good())
		throw std::runtime_error("Could not read CompressedMatrix state");
	if (header.typesize != CSRIOHeader::typeSize<data_t>()) {
		std::cout<< header.typesize << " != " << CSRIOHeader::typeSize<data_t>() << std::endl;
		throw std::runtime_error("File does not contain a CSR matrix with matching type");
	}

	auto res = std::make_shared<CSR>();
	res->alloc(header.num_rows, header.num_columns, header.num_non_zeroes);

	fstream.read(reinterpret_cast<char*>(&res->data[0]), res->nnz * sizeof(data_t));
	fstream.read(reinterpret_cast<char*>(&res->col_ids[0]), res->nnz * sizeof(unsigned int));
	fstream.read(reinterpret_cast<char*>(&res->row_offsets[0]), (res->rows+1) * sizeof(unsigned int));

	if (!fstream.good())
		throw std::runtime_error("Could not read CSR matrix data");

	return res;
}


namespace {
	void dealloc(cuCSR& mat) {
		if (mat.col_ids != nullptr)
			cudaFree(mat.col_ids);
		if (mat.data != nullptr)
			cudaFree(mat.data);
		if (mat.row_offsets != nullptr)
			cudaFree(mat.row_offsets);
		mat.col_ids = nullptr;
		mat.data = nullptr;
		mat.row_offsets = nullptr;
		mat.nnz = 0;
		mat.rows = 0;
	}

	void dealloc(cuCSR& mat, cudaStream_t stream) {
		if (mat.col_ids != nullptr)
			cudaFreeAsync(mat.col_ids, stream);
		if (mat.data != nullptr)
			cudaFreeAsync(mat.data, stream);
		if (mat.row_offsets != nullptr)
			cudaFreeAsync(mat.row_offsets, stream);
		mat.col_ids = nullptr;
		mat.data = nullptr;
		mat.row_offsets = nullptr;
		mat.nnz = 0;
		mat.rows = 0;
	}
}

void cuCSR::alloc(size_t r, size_t c, size_t n, bool allocOffsets)
{
	dealloc(*this);
	rows = r;
	cols = c;
	nnz = n;
	cudaMalloc(&data, sizeof(data_t)*n);
	cudaMalloc(&col_ids, sizeof(index_t)*n);
	if (allocOffsets)
		cudaMalloc(&row_offsets, sizeof(index_t)*(r+1));
	// there is no guarantee that the last element is correct. Use nnz instead.
}

void cuCSR::alloc(size_t r, size_t c, size_t n, bool allocOffsets, cudaStream_t stream)
{
	dealloc(*this, stream);
	rows = r;
	cols = c;
	nnz = n;
	cudaMallocAsync(&data, sizeof(data_t)*n, stream);
	cudaMallocAsync(&col_ids, sizeof(index_t)*n, stream);
	if (allocOffsets)
		cudaMallocAsync(&row_offsets, sizeof(index_t)*(r+1), stream);
	// there is no guarantee that the last element is correct. Use nnz instead.
}

cuCSR::~cuCSR()
{
	dealloc(*this);
}

void cuCSR::reset()
{
	dealloc(*this);
}

void cuCSR::reset(cudaStream_t stream)
{
	dealloc(*this, stream);
}

void convert(cuCSR& dst, const CSR& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8*padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;

	cudaMemcpy(dst.data, &src.data[0], src.nnz * sizeof(data_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.col_ids, &src.col_ids[0], src.nnz * sizeof(index_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.row_offsets, &src.row_offsets[0], (src.rows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);

	if (padding)
	{
		cudaMemset(dst.data + src.nnz, 0, 8 * padding * sizeof(data_t));
		cudaMemset(dst.col_ids + src.nnz, 0, 8 * padding * sizeof(index_t));
		cudaMemset(dst.row_offsets + src.rows + 1, 0, padding * sizeof(index_t));
	}
}

void convert(CSR& dst, const cuCSR& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data.get(), src.data, dst.nnz * sizeof(data_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst.col_ids.get(), src.col_ids, dst.nnz * sizeof(index_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst.row_offsets.get(), src.row_offsets, (dst.rows + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
}

void compare(const CSR& a, const CSR& b)
{
	std::cout << BANNER << std::endl;
	std::cout << "Comparing CSR matrices" << std::endl;

	if (a.rows != b.rows || a.cols != b.cols || a.nnz != b.nnz) {
		std::cout << "a.rows: " << a.rows << ", b.rows: " << b.rows << std::endl;
		std::cout << "a.cols: " << a.cols << ", b.cols: " << b.cols << std::endl;
		std::cout << "a.nnz: " << a.nnz << ", b.nnz: " << b.nnz << std::endl;
        throw std::runtime_error("Matrices have different dimensions");
	}

	for (size_t i = 0; i < a.rows; ++i)
	{
		if (a.row_offsets[i] != b.row_offsets[i])
		{
			std::cout << "Row " << i << " has different row offsets" << std::endl;
			std::cout << "a: " << a.row_offsets[i] << " b: " << b.row_offsets[i] << std::endl;
			throw std::runtime_error("Matrices have different row offsets");
		}
	}

	for (size_t i = 0; i < a.nnz; ++i)
	{
		if (a.col_ids[i] != b.col_ids[i])
		{
			std::cout << "Entry " << i << " has different column IDs" << std::endl;
			std::cout << "a: " << a.col_ids[i] << " b: " << b.col_ids[i] << std::endl;
			// locate the row
			size_t row = 0;
			while (row < a.rows && a.row_offsets[row + 1] <= i)
				++row;
			std::cout << "This is in row " << row << std::endl;

			// print the 10 elements around it
			size_t start = (i < 5) ? 0 : i - 5;
			size_t end = std::min(i + 5, a.nnz - 1);
			std::cout << "a: ";
			for (size_t j = start; j <= end; ++j)
				std::cout << "(" << a.col_ids[j] << ", " << a.data[j] << ") ";
			std::cout << std::endl;
			std::cout << "b: ";
			for (size_t j = start; j <= end; ++j)
				std::cout << "(" << b.col_ids[j] << ", " << b.data[j] << ") ";
			std::cout << std::endl;
			throw std::runtime_error("Matrices have different column IDs");
		}
	}

	// compare data but allow for some error
	// calculate the relative error

	size_t num_errors = 0;
	for (size_t i = 0; i < a.nnz; ++i)
	{
		double relative_error = fabs(a.data[i] - b.data[i]) / std::max(fabs(a.data[i]), fabs(b.data[i]));
		if (relative_error > COMPARE_RELATIVE_ERROR)
		{
			// std::cout << "Entry " << i << " has different data" << std::endl;
			// std::cout << "a: " << a.data[i] << " b: " << b.data[i] << " relative error: " << relative_error << std::endl;
			++num_errors;
		}
		if (relative_error > COMPARE_RELATIVE_ERROR_NO_TOLERATE)
		{
			std::cout << "Entry " << i << " has different data" << std::endl;
			std::cout << "a: " << a.data[i] << " b: " << b.data[i] << " relative error: " << relative_error << std::endl;
			// locate the row
			size_t row = 0;
			while (row < a.rows && a.row_offsets[row + 1] <= i)
				++row;
			std::cout << "This is in row " << row << ", col: " << a.col_ids[i] << std::endl;
			throw std::runtime_error("Matrices have different data");
		}
	}

	std::cout << "No element has relative error greater than " << COMPARE_RELATIVE_ERROR << std::endl;

	double percent_error = 100.0 * num_errors / a.nnz;
	std::cout << num_errors << " / " << a.nnz << " = " << percent_error << "\% elements have relative error greater than " << COMPARE_RELATIVE_ERROR << std::endl;
	std::cout << "Comparison completed" << std::endl;
	std::cout << BANNER << std::endl;

	return;
}

}