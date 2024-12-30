#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <type_traits>
#include <random>  // Requires C++ 11

#include <Spectra/HermEigsSolver.h>
#include <Spectra/MatOp/DenseHermMatProd.h>
#include <Spectra/MatOp/SparseHermMatProd.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<std::complex<double>>;
using RealMatrix = Eigen::MatrixXd;
using RealVector = Eigen::VectorXd;

// Generate data for testing
Matrix gen_dense_data(int n)
{
    const Matrix mat = Matrix::Random(n, n);
    return mat + mat.adjoint();
}

SpMatrix gen_sparse_data(int n, double prob = 0.5)
{
    // Eigen solver only uses the lower triangular part of mat,
    // so we don't need to make mat an Hermitian matrix here,
    // but diagonal elements must have a zero imaginary part
    SpMatrix mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
            {
                double re = distr(gen) - 0.5;
                double im = (i == j) ? 0.0 : (distr(gen) - 0.5);
                mat.insert(i, j) = std::complex<double>(re, im);
            }
        }
    }
    return mat;
}

template <typename MatType, typename Solver>
void run_test(const MatType& mat, Solver& eigs, SortRule selection)
{
    eigs.init();
    int nconv = eigs.compute(selection);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    INFO("nops  = " << nops);
    REQUIRE(eigs.info() == CompInfo::Successful);

    RealVector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    Matrix resid = mat.template selfadjointView<Eigen::Lower>() * evecs - evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

template <typename MatType>
void run_test_sets(const MatType& mat, int k, int m)
{
    constexpr bool is_dense = std::is_same<MatType, Matrix>::value;
    using DenseOp = DenseHermMatProd<std::complex<double>>;
    using SparseOp = SparseHermMatProd<std::complex<double>>;
    using OpType = typename std::conditional<is_dense, DenseOp, SparseOp>::type;

    OpType op(mat);
    HermEigsSolver<OpType> eigs(op, k, m);

    SECTION("Largest Magnitude")
    {
        run_test(mat, eigs, SortRule::LargestMagn);
    }
    SECTION("Largest Value")
    {
        run_test(mat, eigs, SortRule::LargestAlge);
    }
    SECTION("Smallest Magnitude")
    {
        run_test(mat, eigs, SortRule::SmallestMagn);
    }
    SECTION("Smallest Value")
    {
        run_test(mat, eigs, SortRule::SmallestAlge);
    }
    SECTION("Both Ends")
    {
        run_test(mat, eigs, SortRule::BothEnds);
    }
}

TEST_CASE("Eigensolver of Hermitian matrix [10x10]", "[eigs_herm]")
{
    std::srand(123);

    const Matrix A = gen_dense_data(10);
    int k = 3;
    int m = 6;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of Hermitian matrix [100x100]", "[eigs_herm]")
{
    std::srand(123);

    const Matrix A = gen_dense_data(100);
    int k = 10;
    int m = 20;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of Hermitian matrix [1000x1000]", "[eigs_herm]")
{
    std::srand(123);

    const Matrix A = gen_dense_data(1000);
    int k = 20;
    int m = 50;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of sparse Hermitian matrix [10x10]", "[eigs_herm]")
{
    std::srand(123);

    // Eigen solver only uses the lower triangle
    const SpMatrix A = gen_sparse_data(10, 0.5);
    int k = 3;
    int m = 6;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of sparse Hermitian matrix [100x100]", "[eigs_herm]")
{
    std::srand(123);

    // Eigen solver only uses the lower triangle
    const SpMatrix A = gen_sparse_data(100, 0.1);
    int k = 10;
    int m = 20;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of sparse Hermitian matrix [1000x1000]", "[eigs_herm]")
{
    std::srand(123);

    // Eigen solver only uses the lower triangle
    const SpMatrix A = gen_sparse_data(1000, 0.01);
    int k = 20;
    int m = 50;

    run_test_sets(A, k, m);
}
