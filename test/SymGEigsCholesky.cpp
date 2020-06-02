#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <random>  // Requires C++ 11

#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SpMatrix = Eigen::SparseMatrix<double>;

// Traits to obtain operation type from matrix type
template <typename MatType>
struct OpTypeTrait
{
    using OpType = DenseSymMatProd<double>;
};

template <>
struct OpTypeTrait<SpMatrix>
{
    using OpType = SparseSymMatProd<double>;
};

template <typename MatType>
struct BOpTypeTrait
{
    using OpType = DenseCholesky<double>;
};

template <>
struct BOpTypeTrait<SpMatrix>
{
    using OpType = SparseCholesky<double>;
};

// Generate random sparse matrix
SpMatrix sprand(int size, double prob = 0.5)
{
    SpMatrix mat(size, size);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = distr(gen) - 0.5;
        }
    }
    return mat;
}

// Generate data for testing
void gen_dense_data(int n, Matrix& A, Matrix& B)
{
    Matrix M = Eigen::MatrixXd::Random(n, n);
    A = M + M.transpose();
    B = M.transpose() * M;
    // To make sure B is positive definite
    B.diagonal() += Eigen::VectorXd::Random(n).cwiseAbs();
}

void gen_sparse_data(int n, SpMatrix& A, SpMatrix& B, double prob = 0.1)
{
    // Eigen solver only uses the lower triangle of A,
    // so we don't need to make A symmetric here.
    A = sprand(n, prob);
    B = A.transpose() * A;
    // To make sure B is positive definite
    for (int i = 0; i < n; i++)
        B.coeffRef(i, i) += 0.1;
}

template <typename MatType>
void run_test(const MatType& A, const MatType& B, int k, int m, SortRule selection, bool allow_fail = false)
{
    using OpType = typename OpTypeTrait<MatType>::OpType;
    using BOpType = typename BOpTypeTrait<MatType>::OpType;
    OpType op(A);
    BOpType Bop(B);
    // Make sure B is positive definite and the decomposition is successful
    REQUIRE(Bop.info() == CompInfo::Successful);

    SymGEigsSolver<double, OpType, BOpType, GEigsMode::Cholesky> eigs(op, Bop, k, m);
    eigs.init();
    // maxit = 300 to reduce running time for failed cases
    int nconv = eigs.compute(selection, 300);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    if (allow_fail && eigs.info() != CompInfo::Successful)
    {
        WARN("FAILED on this test");
        std::cout << "nconv = " << nconv << std::endl;
        std::cout << "niter = " << niter << std::endl;
        std::cout << "nops  = " << nops << std::endl;
        return;
    }
    else
    {
        INFO("nconv = " << nconv);
        INFO("niter = " << niter);
        INFO("nops  = " << nops);
        REQUIRE(eigs.info() == CompInfo::Successful);
    }

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    Matrix resid = A.template selfadjointView<Eigen::Lower>() * evecs -
        B.template selfadjointView<Eigen::Lower>() * evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    INFO("||AU - BUD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

template <typename MatType>
void run_test_sets(const MatType& A, const MatType& B, int k, int m)
{
    SECTION("Largest Magnitude")
    {
        run_test<MatType>(A, B, k, m, SortRule::LargestMagn);
    }
    SECTION("Largest Value")
    {
        run_test<MatType>(A, B, k, m, SortRule::LargestAlge);
    }
    SECTION("Smallest Magnitude")
    {
        run_test<MatType>(A, B, k, m, SortRule::SmallestMagn, true);
    }
    SECTION("Smallest Value")
    {
        run_test<MatType>(A, B, k, m, SortRule::SmallestAlge);
    }
    SECTION("Both Ends")
    {
        run_test<MatType>(A, B, k, m, SortRule::BothEnds);
    }
}

TEST_CASE("Generalized eigensolver of symmetric real matrix [10x10]", "[geigs_sym]")
{
    std::srand(123);

    Matrix A, B;
    gen_dense_data(10, A, B);
    int k = 3;
    int m = 6;

    run_test_sets(A, B, k, m);
}

TEST_CASE("Generalized eigensolver of symmetric real matrix [100x100]", "[geigs_sym]")
{
    std::srand(123);

    Matrix A, B;
    gen_dense_data(100, A, B);
    int k = 10;
    int m = 20;

    run_test_sets(A, B, k, m);
}

TEST_CASE("Generalized eigensolver of symmetric real matrix [1000x1000]", "[geigs_sym]")
{
    std::srand(123);

    Matrix A, B;
    gen_dense_data(1000, A, B);
    int k = 20;
    int m = 50;

    run_test_sets(A, B, k, m);
}

TEST_CASE("Generalized eigensolver of sparse symmetric real matrix [10x10]", "[geigs_sym]")
{
    std::srand(123);

    // Eigen solver only uses the lower triangle
    SpMatrix A, B;
    gen_sparse_data(10, A, B, 0.5);
    int k = 3;
    int m = 6;

    run_test_sets(A, B, k, m);
}

TEST_CASE("Generalized eigensolver of sparse symmetric real matrix [100x100]", "[geigs_sym]")
{
    std::srand(123);

    // Eigen solver only uses the lower triangle
    SpMatrix A, B;
    gen_sparse_data(100, A, B);
    int k = 10;
    int m = 20;

    run_test_sets(A, B, k, m);
}

TEST_CASE("Generalized eigensolver of sparse symmetric real matrix [1000x1000]", "[geigs_sym]")
{
    std::srand(123);

    // Eigen solver only uses the lower triangle
    SpMatrix A, B;
    gen_sparse_data(1000, A, B);
    int k = 20;
    int m = 50;

    run_test_sets(A, B, k, m);
}
