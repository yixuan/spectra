#include <Eigen/Sparse>
#include <random>

// Generate Matrix
Eigen::SparseMatrix<double> gen_sym_data_sparse(int n)
{
    double prob = 0.5;
    Eigen::SparseMatrix<double> mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = distr(gen) - 0.5;
        }
    }
    return mat + Eigen::SparseMatrix<double>(mat.transpose());
}

Eigen::SparseMatrix<double> A = gen_sym_data_sparse(1000);

#include <Spectra/MatOp/SparseSymMatProd.h>

Spectra::SparseSymMatProd<double> op(A); // Create the Matrix Product operation

#include <Spectra/JDSymEigsDPR.h>

Spectra::JDSymEigsDPR<Spectra::SparseSymMatProd<double>> solver(op,2); //Create Solver

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
TEST_CASE("Davidson Symmetric DPR example")
{
    REQUIRE(true);
}