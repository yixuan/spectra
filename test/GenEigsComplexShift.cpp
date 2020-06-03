#include <Eigen/Core>
#include <iostream>

#include <Spectra/GenEigsComplexShiftSolver.h>
#include <Spectra/MatOp/DenseGenComplexShiftSolve.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;

void run_test(const Matrix &mat, int k, int m, double sigmar, double sigmai,
              SortRule selection, bool allow_fail = false)
{
    DenseGenComplexShiftSolve<double> op(mat);
    GenEigsComplexShiftSolver<double, DenseGenComplexShiftSolve<double>> eigs(op, k, m, sigmar, sigmai);
    eigs.init();
    // maxit = 100 to reduce running time for failed cases
    int nconv = eigs.compute(selection, 100);
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
    INFO("||AU - UD||_2 colwise =" << resid.colwise().norm());
    REQUIRE(err == Approx(0.0).margin(1e-8));
}

void run_test_sets(const Matrix &A, int k, int m, double sigmar, double sigmai)
{
    SECTION("Largest Magnitude")
    {
        run_test(A, k, m, sigmar, sigmai, SortRule::LargestMagn);
    }
    SECTION("Largest Real Part")
    {
        run_test(A, k, m, sigmar, sigmai, SortRule::LargestReal);
    }
    SECTION("Largest Imaginary Part")
    {
        run_test(A, k, m, sigmar, sigmai, SortRule::LargestImag);
    }
    SECTION("Smallest Magnitude")
    {
        run_test(A, k, m, sigmar, sigmai, SortRule::SmallestMagn, true);
    }
    SECTION("Smallest Real Part")
    {
        run_test(A, k, m, sigmar, sigmai, SortRule::SmallestReal);
    }
    SECTION("Smallest Imaginary Part")
    {
        run_test(A, k, m, sigmar, sigmai, SortRule::SmallestImag, true);
    }
}

TEST_CASE("Eigensolver of general real matrix [10x10]", "[eigs_gen]")
{
    std::srand(123);

    Matrix A = Eigen::MatrixXd::Random(10, 10);
    int k = 3;
    int m = 8;
    double sigmar = 2.0;
    double sigmai = 1.0;

    run_test_sets(A, k, m, sigmar, sigmai);
}

TEST_CASE("Eigensolver of general real matrix [100x100]", "[eigs_gen]")
{
    std::srand(123);

    Matrix A = Eigen::MatrixXd::Random(100, 100);
    int k = 10;
    int m = 20;
    double sigmar = 20.0;
    double sigmai = 10.0;

    run_test_sets(A, k, m, sigmar, sigmai);
}

TEST_CASE("Eigensolver of general real matrix [1000x1000]", "[eigs_gen]")
{
    std::srand(123);

    Matrix A = Eigen::MatrixXd::Random(1000, 1000);
    int k = 20;
    int m = 50;
    double sigmar = 200.0;
    double sigmai = 100.0;

    run_test_sets(A, k, m, sigmar, sigmai);
}
