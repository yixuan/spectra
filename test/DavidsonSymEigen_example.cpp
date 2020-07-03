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

// Maximum size of the search space
solver.setMaxSearchSpaceSize(250); 

// Number of corretion vector to append to the
// search space at each iteration
solver.setCorrectionSize(4);

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
TEST_CASE("Davidson Symmetric EigenSolver example")
{
    solver.compute(Spectra::SortRule::LargestAlge, maxit = 100, tol=1E-3);
    REQUIRE(solve.info() == CompInfo::Successful);
}

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
TEST_CASE("Davidson Symmetric EigenSolver example with guess")
{
    Matrix guess = Eigen::Random(1000, 4);
    Spectra::QR_orthogonalisation(guess);
    solver.computeWithGuess(guess, Spectra::SortRule::LargestAlge, maxit=100, tol=1E-3);
    REQUIRE(solve.info() == CompInfo::Successful);
}
