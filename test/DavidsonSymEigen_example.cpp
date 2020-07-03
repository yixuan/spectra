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
            if (i == j)
                mat.coeffRef(i, j) = 10 * n;
        }
    }
    return mat + Eigen::SparseMatrix<double>(mat.transpose());
}

Eigen::SparseMatrix<double> A = gen_sym_data_sparse(1000);

#include <Spectra/MatOp/SparseSymMatProd.h>

Spectra::SparseSymMatProd<double> op(A); // Create the Matrix Product operation

#include <Spectra/DavidsonSymEig.h>

Spectra::DavidsonSymEig<Spectra::SparseSymMatProd<double>> solver(op,2); //Create Solver

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
TEST_CASE("Davidson Symmetric EigenSolver example")
{
    // Maximum size of the search space
    solver.setMaxSearchspaceSize(250); 

    // Number of corretion vector to append to the
    // search space at each iteration
    solver.setCorrectionSize(4);

    Eigen::MatrixXd guess = Eigen::MatrixXd::Random(1000, 4);
    Spectra::QR_orthogonalisation(guess);
    solver.computeWithGuess(guess, Spectra::SortRule::LargestAlge, 100, 1E-3);
    REQUIRE(solver.info() == Spectra::CompInfo::Successful);
}

#include <Eigen/Dense>

// Generate data for testing
Eigen::MatrixXd gen_sym_data_dense(int n)
{
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd mat1 = mat + mat.transpose();
    mat1.diagonal().array() = 10* n;
    return mat1;
}

Eigen::MatrixXd B = gen_sym_data_dense(1000);

#include <Spectra/MatOp/DenseSymMatProd.h>

Spectra::DenseSymMatProd<double> op_dense(B); // Create the Matrix Product operation

Spectra::DavidsonSymEig<Spectra::DenseSymMatProd<double>> solver_dense(op_dense,2); //Create Solver

TEST_CASE("Davidson Dense Symmetric EigenSolver example")
{
    Eigen::MatrixXd guess = Eigen::MatrixXd::Random(1000, 4);
    Spectra::QR_orthogonalisation(guess);
    solver_dense.computeWithGuess(guess, Spectra::SortRule::LargestAlge, 100, 1E-3);
    REQUIRE(solver_dense.info() == Spectra::CompInfo::Successful);
}