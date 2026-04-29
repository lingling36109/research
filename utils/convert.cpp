// Standalone converter from .mtx format to Speck's CSR binary format (.hicsr)
// Adapted from Speck's repository by Claude

#include <stdint.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iterator>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>
#include <sstream>
#include <cstring>

// ==================== Data Structures ====================

template<typename T>
struct COO
{
	size_t rows, cols, nnz;
	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_ids;
	std::unique_ptr<unsigned int[]> col_ids;

	COO() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t r, size_t c, size_t n)
	{
		rows = r;
		cols = c;
		nnz = n;
		data = std::make_unique<T[]>(n);
		row_ids = std::make_unique<unsigned int[]>(n);
		col_ids = std::make_unique<unsigned int[]>(n);
	}
};

template<typename T>
struct CSR
{
	size_t rows, cols, nnz;
	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_offsets;
	std::unique_ptr<unsigned int[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t r, size_t c, size_t n)
	{
		rows = r;
		cols = c;
		nnz = n;
		data = std::make_unique<T[]>(n);
		col_ids = std::make_unique<unsigned int[]>(n);
		row_offsets = std::make_unique<unsigned int[]>(r+1);
	}
};

// ==================== MTX Loading ====================

namespace {
	template<typename VALUE_TYPE>
	struct DataTypeValidator {
		static const bool validate(std::string type) {
			return false;
		}
	};

	template<>
	struct DataTypeValidator<float> {
		static const bool validate(std::string type) {
			return type.compare("real") == 0 || type.compare("integer") == 0 || type.compare("double") == 0;
		}
	};
	
	template<>
	struct DataTypeValidator<double> {
		static const bool validate(std::string type) {
			return type.compare("real") == 0 || type.compare("integer") == 0 || type.compare("double") == 0;
		}
	};
}

template<typename T>
COO<T> loadMTX(const char * file)
{
	std::ifstream fstream(file);
	if (!fstream.is_open())
		throw std::runtime_error(std::string("could not open \"") + file + "\"");
	
	COO<T> resmatrix;
	size_t num_rows, num_columns, num_non_zeroes;

	size_t line_counter = 0;
	std::string line;
	bool pattern = false;
	bool hermitian = false;
	
	// read header
	std::getline(fstream, line);
	if (line.compare(0, 32, "%%MatrixMarket matrix coordinate") != 0)
		throw std::runtime_error("Can only read MatrixMarket format that is in coordinate form");
	
	std::istringstream iss(line);
	std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{} };
	bool complex = false;

	if (tokens[3] == "pattern")
		pattern = true;
	else if (tokens[3] == "complex")
		complex = true;
	else if (DataTypeValidator<T>::validate(tokens[3]) == false)
		throw std::runtime_error("MatrixMarket data type does not match matrix format");
	
	bool symmetric = false;
	if (tokens[4].compare("general") == 0)
		symmetric = false;
	else if (tokens[4].compare("symmetric") == 0)
		symmetric = true;
	else if (tokens[4].compare("Hermitian") == 0)
		hermitian = true;
	else
		throw std::runtime_error("Can only read MatrixMarket format that is either symmetric, general or hermitian");

	while (std::getline(fstream, line))
	{
		++line_counter;
		if (line[0] == '%')
			continue;
		std::istringstream liness(line);
		liness >> num_rows >> num_columns >> num_non_zeroes;
		if (liness.fail())
			throw std::runtime_error(std::string("Failed to read matrix market header from \"") + file + "\"");
		break;
	}

	size_t reserve = num_non_zeroes;
	if (symmetric || hermitian)
		reserve *= 2;

	resmatrix.alloc(num_rows, num_columns, reserve);

	// read data
	size_t read = 0;
	while (std::getline(fstream, line))
	{
		++line_counter;
		if (line[0] == '%')
			continue;

		std::istringstream liness(line);

		do
		{
			char ch;
			liness.get(ch);
			if (!isspace(ch))
			{
				liness.putback(ch);
				break;
			}
		} while (!liness.eof());
		
		if (liness.eof() || line.length() == 0)
			continue;

		uint32_t r, c;
		T d;
		liness >> r >> c;
		if (pattern)
			d = 1;
		else
			liness >> d;
		
		if (liness.fail())
			throw std::runtime_error(std::string("Failed to read data at line ") + std::to_string(line_counter) + " from matrix market file \"" + file + "\"");
		if (r > num_rows)
			throw std::runtime_error(std::string("Row index out of bounds at line  ") + std::to_string(line_counter) + " in matrix market file \"" + file + "\"");
		if (c > num_columns)
			throw std::runtime_error(std::string("Column index out of bounds at line  ") + std::to_string(line_counter) + " in matrix market file \"" + file + "\"");
		
		resmatrix.row_ids[read] = r - 1;
		resmatrix.col_ids[read] = c - 1;
		resmatrix.data[read] = d;
		++read;
		
		if ((symmetric || hermitian) && r != c)
		{
			resmatrix.row_ids[read] = c - 1;
			resmatrix.col_ids[read] = r - 1;
			resmatrix.data[read] = d;
			++read;
		}
	}

	resmatrix.nnz = read;
	return resmatrix;
}

// ==================== COO to CSR Conversion ====================

template<typename T>
void convert(CSR<T>& res, const COO<T>& coo)
{
	struct Entry
	{
		unsigned int r, c;
		T v;
		bool operator < (const Entry& other)
		{
			if (r != other.r) 
				return r < other.r;
			return c < other.c;
		}
	};

	std::vector<Entry> entries;
	entries.reserve(coo.nnz);
	for (size_t i = 0; i < coo.nnz; ++i)
		entries.push_back(Entry{ coo.row_ids[i], coo.col_ids[i], coo.data[i] });
	std::sort(std::begin(entries), std::end(entries));

	res.alloc(coo.rows, coo.cols, coo.nnz);
	std::fill(&res.row_offsets[0], &res.row_offsets[coo.rows], 0);
	
	for (size_t i = 0; i < coo.nnz; ++i)
	{
		res.data[i] = entries[i].v;
		res.col_ids[i] = entries[i].c;
		++res.row_offsets[entries[i].r];
	}

	unsigned int off = 0;
	for (size_t i = 0; i < coo.rows; ++i)
	{
		unsigned int n = off + res.row_offsets[i];
		res.row_offsets[i] = off;
		off = n;
	}
	res.row_offsets[coo.rows] = off;
}

// ==================== CSR Storage (Speck's .hicsr format) ====================

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
		CSRIOHeader(const CSR<T>& mat)
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
	};
	constexpr char CSRIOHeader::Magic[];
}

template<typename T>
void storeCSR(const CSR<T>& mat, const char * file)
{
	std::ofstream fstream(file, std::fstream::binary);
	if (!fstream.is_open())
		throw std::runtime_error(std::string("could not open \"") + file + "\"");

	CSRIOHeader header(mat);
	State<T> state;
	fstream.write(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));
	fstream.write(reinterpret_cast<const char*>(&state), sizeof(state));
	fstream.write(reinterpret_cast<char*>(&mat.data[0]), mat.nnz * sizeof(T));
	fstream.write(reinterpret_cast<char*>(&mat.col_ids[0]), mat.nnz * sizeof(unsigned int));
	fstream.write(reinterpret_cast<char*>(&mat.row_offsets[0]), (mat.rows + 1) * sizeof(unsigned int));
}

// ==================== Main ====================

template<typename T>
void convertMTXtoCSR(const char* input_file, const char* output_file)
{
	std::cout << "Loading MTX file: " << input_file << std::endl;
	COO<T> coo = loadMTX<T>(input_file);
	std::cout << "  Matrix dimensions: " << coo.rows << " x " << coo.cols << std::endl;
	std::cout << "  Non-zeros: " << coo.nnz << std::endl;
	
	std::cout << "Converting to CSR format..." << std::endl;
	CSR<T> csr;
	convert(csr, coo);
	
	std::cout << "Writing CSR to file: " << output_file << std::endl;
	storeCSR(csr, output_file);
	
	std::cout << "Conversion complete!" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " <input.mtx> <output.csr> [float|double]" << std::endl;
		std::cout << "  Default data type is double if not specified" << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* output_file = argv[2];
	std::string data_type = (argc > 3) ? argv[3] : "double";

	try
	{
		if (data_type == "float")
		{
			convertMTXtoCSR<float>(input_file, output_file);
		}
		else if (data_type == "double")
		{
			convertMTXtoCSR<double>(input_file, output_file);
		}
		else
		{
			std::cerr << "Invalid data type: " << data_type << std::endl;
			std::cerr << "Supported types: float, double" << std::endl;
			return 1;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
