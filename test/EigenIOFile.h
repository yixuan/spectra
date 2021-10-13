
#ifndef EIGENIOFILE_H
#define EIGENIOFILE_H

// Reference: https://gist.github.com/zishun/da277d30f4604108029d06db0e804773

#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Eigen {
    // https://stackoverflow.com/a/25389481/11927397
    template<class Matrix>
    inline void write_binary(const std::string& filename, const Matrix& matrix) {
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        if (out.is_open()) {
            typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
            out.write(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
            out.write(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
            out.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)));
            out.close();
        }
        else {
            std::cout << "Cannot write to file: " << filename << std::endl;
        }
    }

    template<class Matrix>
    inline void read_binary(const std::string& filename, Matrix& matrix) {
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        if (in.is_open()) {
            typename Matrix::Index rows = 0, cols = 0;
            in.read(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
            in.read(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
            matrix.resize(rows, cols);
            in.read(reinterpret_cast<char*>(matrix.data()), rows * cols * static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)));
            in.close();
        }
        else {
            std::cout << "Cannot open binary matrix file: " << filename << std::endl;
        }
    }

    // https://scicomp.stackexchange.com/a/21438
    template <class SparseMatrix>
    inline void write_binary_sparse(const std::string& filename, const SparseMatrix& matrix) {
        assert(matrix.isCompressed() == true);
        std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
        if (out.is_open())
        {
            typename SparseMatrix::Index rows, cols, nnzs, outS, innS;
            rows = matrix.rows();
            cols = matrix.cols();
            nnzs = matrix.nonZeros();
            outS = matrix.outerSize();
            innS = matrix.innerSize();

            out.write(reinterpret_cast<char*>(&rows), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&cols), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&nnzs), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&outS), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&innS), sizeof(typename SparseMatrix::Index));

            typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
            typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar));
            out.write(reinterpret_cast<const char*>(matrix.valuePtr()), sizeScalar * nnzs);
            out.write(reinterpret_cast<const char*>(matrix.outerIndexPtr()), sizeIndexS * outS);
            out.write(reinterpret_cast<const char*>(matrix.innerIndexPtr()), sizeIndexS * nnzs);

            out.close();
        }
        else {
            std::cout << "Cannot write to file: " << filename << std::endl;
        }
    }

    template <class SparseMatrix>
    inline void read_binary_sparse(const std::string& filename, SparseMatrix& matrix) {
        std::ifstream in(filename, std::ios::binary | std::ios::in);
        if (in.is_open()) {
            typename SparseMatrix::Index rows, cols, nnz, inSz, outSz;
            typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar));
            typename SparseMatrix::Index sizeIndex = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Index));
            typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
            std::cout << sizeScalar << " " << sizeIndex << std::endl;
            in.read(reinterpret_cast<char*>(&rows), sizeIndex);
            in.read(reinterpret_cast<char*>(&cols), sizeIndex);
            in.read(reinterpret_cast<char*>(&nnz), sizeIndex);
            in.read(reinterpret_cast<char*>(&outSz), sizeIndex);
            in.read(reinterpret_cast<char*>(&inSz), sizeIndex);

            matrix.resize(rows, cols);
            matrix.makeCompressed();
            matrix.resizeNonZeros(nnz);

            in.read(reinterpret_cast<char*>(matrix.valuePtr()), sizeScalar * nnz);
            in.read(reinterpret_cast<char*>(matrix.outerIndexPtr()), sizeIndexS * outSz);
            in.read(reinterpret_cast<char*>(matrix.innerIndexPtr()), sizeIndexS * nnz);

            matrix.finalize();
            in.close();
        } // file is open
        else {
            std::cout << "Cannot open binary sparse matrix file: " << filename << std::endl;
        }
    }
} // Eigen::

#endif // EIGENIOFILE_H