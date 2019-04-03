// Test ../include/Spectra/LinAlg/BKLDLT.h
#include <Eigen/Core>
#include <Spectra/LinAlg/BKLDLT.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void run_test(const MatrixXd& A, const VectorXd& b)
{
    BKLDLT<double> decompL(A, Eigen::Lower);
    REQUIRE(decompL.info() == SUCCESSFUL);

    BKLDLT<double> decompU(A, Eigen::Upper);
    REQUIRE( decompU.info() == SUCCESSFUL );

    VectorXd solL = decompL.solve(b);
    VectorXd solU = decompU.solve(b);
    REQUIRE( (solL - solU).cwiseAbs().maxCoeff() == 0.0 );

    const double tol = 1e-10;
    VectorXd resid = A * solL - b;
    INFO( "||A * x - b||_inf = " << resid.cwiseAbs().maxCoeff() );
    REQUIRE( resid.cwiseAbs().maxCoeff() == Approx(0.0).margin(tol) );
}

TEST_CASE("BKLDLT decomposition of symmetric real matrix [10x10]", "[BKLDLT]")
{
    std::srand(123);
    int n = 10;
    MatrixXd A = MatrixXd::Random(n, n);
    A = (A + A.transpose()).eval();
    VectorXd b = VectorXd::Random(n);

    run_test(A, b);
}

TEST_CASE("BKLDLT decomposition of symmetric real matrix [100x100]", "[BKLDLT]")
{
    std::srand(123);
    int n = 100;
    MatrixXd A = MatrixXd::Random(n, n);
    A = (A + A.transpose()).eval();
    VectorXd b = VectorXd::Random(n);

    run_test(A, b);
}

TEST_CASE("BKLDLT decomposition of symmetric real matrix [1000x1000]", "[BKLDLT]")
{
    std::srand(123);
    int n = 1000;
    MatrixXd A = MatrixXd::Random(n, n);
    A = (A + A.transpose()).eval();
    VectorXd b = VectorXd::Random(n);

    run_test(A, b);
}
