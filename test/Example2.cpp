// Example reported in Issue #159
// https://github.com/yixuan/spectra/issues/159
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

void run_test(Matrix& M)
{
    // True eigenvalues
    Eigen::SelfAdjointEigenSolver<Matrix> es(M);
    Vector true_evals = es.eigenvalues();

    // Largest eigenvalues
    DenseSymMatProd<double> op(M);
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, 1, 3);

    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn);
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
    REQUIRE(err == Approx(0.0).margin(1e-8));

    INFO("True eigenvalues =\n " << true_evals);
    INFO("Estimated =\n " << evals);
    double diff = (true_evals.tail(1) - evals).array().abs().maxCoeff();
    INFO("diff = " << diff);
    REQUIRE(diff == Approx(0.0).margin(1e-8));
}

TEST_CASE("Example #159, case 1", "[example_159]")
{
    Matrix M(5, 5);
    M << 15.035447086947079479, 3.932587856183598677, -4.848070276813470542, -8.027254936523050904, -2.865327349780228231,
        3.932587856183598677, 1.028585791773944732, -1.268034278346991263, -2.099564123322002035, -0.749439073848281425,
        -4.848070276813470542, -1.268034278346991263, 1.563224909309606855, 2.588329820664053864, 0.923903910371237535,
        -8.027254936523050904, -2.099564123322002035, 2.588329820664053864, 4.285660509016328222, 1.529765824738644411,
        -2.865327349780228231, -0.749439073848281425, 0.923903910371237535, 1.529765824738644411, 0.546049663433429209;

    run_test(M);
}

TEST_CASE("Example #159, case 2", "[example_159]")
{
    Matrix M(5, 5);
    M << 0.6118330552, -3.058379358, 1.329013596, 2.601267208, 1.072783220,
        -3.058379358, 15.28796821, -6.643360824, -13.00299463, -5.362538075,
        1.329013596, -6.643360824, 2.886861251, 5.650429406, 2.330281884,
        2.601267208, -13.00299463, 5.650429406, 11.05953826, 4.561041261,
        1.072783220, -5.362538075, 2.330281884, 4.561041261, 1.881009576;
    run_test(M);
}

TEST_CASE("Example #159, case 3", "[example_159]")
{
    Matrix M(5, 5);
    M << 17.7699571312182, 10.7033479738827, -19.1658731825582, -4.20053658859459, -11.1426294187651,
        10.7033479738827, 6.44692933157151, -11.5441477084849, -2.53010203979439, -6.71152097511499,
        -19.1658731825582, -11.5441477084849, 20.6714451890590, 4.53050904744533, 12.0179368348118,
        -4.20053658859459, -2.53010203979439, 4.53050904744533, 0.992940360059961, 2.63394122006329,
        -11.1426294187651, -6.71152097511499, 12.0179368348118, 2.63394122006329, 6.98697185632535;

    run_test(M);
}
