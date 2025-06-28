#include <iostream>
#include <type_traits>
#include <random>
#include <complex>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include "catch.hpp"

using namespace Spectra;

using Complex = std::complex<double>;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using ComplexSpMatrix = Eigen::SparseMatrix<Complex>;

ComplexSpMatrix gen_sparse_data(int n, double prob = 0.5)
{
    ComplexSpMatrix mat(n, n);
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
                double im = distr(gen) - 0.5;
                mat.insert(i, j) = Complex(re, im);
            }
        }
    }
    return mat;
}

template <typename MatType, typename Solver>
void run_test(const MatType& mat, Solver& eigs, SortRule selection, bool allow_fail = false)
{
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

    ComplexVector evals = eigs.eigenvalues();
    ComplexMatrix evecs = eigs.eigenvectors();

    ComplexMatrix resid = mat * evecs - evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

template <typename MatType>
void run_test_sets(const MatType& A, int k, int m)
{
    constexpr bool is_dense = std::is_same<MatType, ComplexMatrix>::value;
    using DenseOp = DenseGenMatProd<Complex>;
    using SparseOp = SparseGenMatProd<Complex>;
    using OpType = typename std::conditional<is_dense, DenseOp, SparseOp>::type;

    OpType op(A);
    GenEigsSolver<OpType> eigs(op, k, m);

    SECTION("Largest Magnitude")
    {
        run_test(A, eigs, SortRule::LargestMagn);
    }
    SECTION("Largest Real Part")
    {
        run_test(A, eigs, SortRule::LargestReal);
    }
    SECTION("Largest Imaginary Part")
    {
        run_test(A, eigs, SortRule::LargestImag);
    }
    SECTION("Smallest Magnitude")
    {
        run_test(A, eigs, SortRule::SmallestMagn, true);
    }
    SECTION("Smallest Real Part")
    {
        run_test(A, eigs, SortRule::SmallestReal);
    }
    SECTION("Smallest Imaginary Part")
    {
        run_test(A, eigs, SortRule::SmallestImag, true);
    }
}

TEST_CASE("Eigensolver of general complex matrix [10x10]", "[eigs_complex]")
{
    std::srand(123);

    const ComplexMatrix A = ComplexMatrix::Random(10, 10);
    int k = 3;
    int m = 6;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of general complex matrix [100x100]", "[eigs_complex]")
{
    std::srand(123);

    const ComplexMatrix A = ComplexMatrix::Random(100, 100);
    int k = 10;
    int m = 30;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of general complex matrix [1000x1000]", "[eigs_complex]")
{
    std::srand(123);

    const ComplexMatrix A = ComplexMatrix::Random(1000, 1000);
    int k = 20;
    int m = 50;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of sparse complex matrix [10x10]", "[eigs_complex]")
{
    std::srand(123);

    const ComplexSpMatrix A = gen_sparse_data(10, 0.5);
    int k = 3;
    int m = 6;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of sparse complex matrix [100x100]", "[eigs_complex]")
{
    std::srand(123);

    const ComplexSpMatrix A = gen_sparse_data(100, 0.1);
    int k = 10;
    int m = 30;

    run_test_sets(A, k, m);
}

TEST_CASE("Eigensolver of sparse complex matrix [1000x1000]", "[eigs_complex]")
{
    std::srand(123);

    const ComplexSpMatrix A = gen_sparse_data(1000, 0.01);
    int k = 20;
    int m = 50;

    run_test_sets(A, k, m);
}
