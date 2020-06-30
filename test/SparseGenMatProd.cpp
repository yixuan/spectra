
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/Dense>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::Index;
using SparseMatrixD = Eigen::SparseMatrix<double>;
using triplet = Eigen::Triplet<double>;

SparseMatrixD generate_random_sparse(Index rows, Index cols)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<Eigen::Triplet<double>> tripletVector;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
            auto v_ij = dist(gen);
            if (v_ij < 0.5)
            {
                //if larger than treshold, insert it
                tripletVector.push_back(triplet(i, j, v_ij));
            }
        }
    SparseMatrixD mat(rows, cols);
    //create the matrix
    mat.setFromTriplets(tripletVector.begin(), tripletVector.end());

    return mat;
}

TEST_CASE("matrix operations", "[DenseGenMatProd]")
{
    std::srand(123);
    constexpr Index n = 100;

    SparseMatrixD mat1 = generate_random_sparse(n, n);
    SparseMatrixD mat2 = generate_random_sparse(n, n);

    SparseGenMatProd<double> sparse1(mat1);
    SparseMatrixD xs = sparse1 * mat2;
    SparseMatrixD ys = mat1 * mat2;

    INFO("The matrix-matrix product must be the same as in eigen.")
    REQUIRE(xs.isApprox(ys));
    INFO("The accesor operator must produce the same element as in eigen")
    REQUIRE(mat1.coeff(45, 22) == sparse1(45, 22));
}
