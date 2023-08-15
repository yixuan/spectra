// Example reported in Issue #144
// https://github.com/yixuan/spectra/issues/144
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Matrix construct_cycle_laplacian(int n)
{
    // Initialize the Laplacian matrix
    Matrix L = Matrix::Zero(n, n);

    // Add the matrix entries, iterating over the rows
    for (int i = 0; i < n; i++)
    {
        L(i, i) = 1;
        L(i, (i + n - 1) % n) = -0.5;
        L(i, (i + 1) % n) = -0.5;
    }

    return L;
}

void run_test(int n, int k, int m)
{
    const Matrix M = construct_cycle_laplacian(n);

    // True eigenvalues
    Eigen::SelfAdjointEigenSolver<Matrix> es(M);
    Vector true_evals = es.eigenvalues();

    // Largest eigenvalues
    DenseSymMatProd<double> op(M);
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, k, m);

    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn, 1000, 1e-15, SortRule::SmallestAlge);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    INFO("nops  = " << nops);
    REQUIRE(eigs.info() == CompInfo::Successful);

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();
    Matrix resid = M.selfadjointView<Eigen::Lower>() * evecs - evecs * evals.asDiagonal();
    double err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));

    INFO("True eigenvalues =\n " << true_evals);
    INFO("Estimated =\n " << evals);
    double diff = (true_evals.tail(k) - evals).array().abs().maxCoeff();
    INFO("diff = " << diff);
    REQUIRE(diff == Approx(0.0).margin(1e-9));

    // Smallest eigenvalues
    DenseSymShiftSolve<double> op2(M);
    SymEigsShiftSolver<DenseSymShiftSolve<double>> eigs2(op2, k, m, -1e-6);

    eigs2.init();
    nconv = eigs2.compute(SortRule::LargestMagn, 1000, 1e-15, SortRule::SmallestAlge);
    niter = eigs2.num_iterations();
    nops = eigs2.num_operations();

    INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    INFO("nops  = " << nops);
    REQUIRE(eigs2.info() == CompInfo::Successful);

    evals = eigs2.eigenvalues();
    evecs = eigs2.eigenvectors();
    resid = M.selfadjointView<Eigen::Lower>() * evecs - evecs * evals.asDiagonal();
    err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));

    INFO("Estimated =\n " << evals);
    diff = (true_evals.head(k) - evals).array().abs().maxCoeff();
    INFO("diff = " << diff);
    REQUIRE(diff == Approx(0.0).margin(1e-9));
}

TEST_CASE("Example #144, (n, k, m) = (20, 3, 6)", "[example_144]")
{
    std::srand(123);

    int n = 20;
    int k = 3;
    int m = 6;

    run_test(n, k, m);
}

TEST_CASE("Example #144, (n, k, m) = (20, 5, 12)", "[example_144]")
{
    std::srand(123);

    int n = 20;
    int k = 5;
    int m = 12;

    run_test(n, k, m);
}

TEST_CASE("Example #144, (n, k, m) = (20, 6, 12)", "[example_144]")
{
    std::srand(123);

    int n = 20;
    int k = 6;
    int m = 12;

    run_test(n, k, m);
}
