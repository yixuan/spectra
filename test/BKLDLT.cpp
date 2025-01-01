// Test ../include/Spectra/LinAlg/BKLDLT.h
#include <Eigen/Core>
#include <Spectra/LinAlg/BKLDLT.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;

// Solve (A - s * I)x = b
template <typename MatType, typename VecType>
void run_test(const MatType& A, const VecType& b, double s)
{
    using Scalar = typename MatType::Scalar;

    // Test decomposition using only the lower triangular part
    BKLDLT<Scalar> decompL(A, Eigen::Lower, s);
    REQUIRE(decompL.info() == CompInfo::Successful);

    // Test decomposition using only the upper triangular part
    BKLDLT<Scalar> decompU(A, Eigen::Upper, s);
    REQUIRE(decompU.info() == CompInfo::Successful);

    // Test whether the solutions are identical
    VecType solL = decompL.solve(b);
    VecType solU = decompU.solve(b);
    REQUIRE((solL - solU).cwiseAbs().maxCoeff() == 0.0);

    // Test the accuracy of the solution
    constexpr double tol = 1e-9;
    VecType resid = A * solL - s * solL - b;
    INFO("||(A - s * I)x - b||_inf = " << resid.cwiseAbs().maxCoeff());
    REQUIRE(resid.cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
}

TEST_CASE("BKLDLT decomposition of symmetric real matrix [10x10]", "[BKLDLT]")
{
    std::srand(123);
    const int n = 10;
    Matrix A = Matrix::Random(n, n);
    A = (A + A.transpose()).eval();
    Vector b = Vector::Random(n);
    const double shift = 1.0;

    run_test(A, b, shift);
}

TEST_CASE("BKLDLT decomposition of symmetric real matrix [100x100]", "[BKLDLT]")
{
    std::srand(123);
    const int n = 100;
    Matrix A = Matrix::Random(n, n);
    A = (A + A.transpose()).eval();
    Vector b = Vector::Random(n);
    const double shift = 1.0;

    run_test(A, b, shift);
}

TEST_CASE("BKLDLT decomposition of symmetric real matrix [1000x1000]", "[BKLDLT]")
{
    std::srand(123);
    const int n = 1000;
    Matrix A = Matrix::Random(n, n);
    A = (A + A.transpose()).eval();
    Vector b = Vector::Random(n);
    const double shift = 1.0;

    run_test(A, b, shift);
}

TEST_CASE("BKLDLT decomposition of Hermitian matrix [10x10]", "[BKLDLT]")
{
    std::srand(123);
    const int n = 10;
    ComplexMatrix A = ComplexMatrix::Random(n, n);
    A = (A + A.adjoint()).eval();
    ComplexVector b = ComplexVector::Random(n);
    const double shift = 1.0;

    run_test(A, b, shift);
}

TEST_CASE("BKLDLT decomposition of Hermitian matrix [100x100]", "[BKLDLT]")
{
    std::srand(123);
    const int n = 100;
    ComplexMatrix A = ComplexMatrix::Random(n, n);
    A = (A + A.adjoint()).eval();
    ComplexVector b = ComplexVector::Random(n);
    const double shift = 1.0;

    run_test(A, b, shift);
}

TEST_CASE("BKLDLT decomposition of Hermitian matrix [1000x1000]", "[BKLDLT]")
{
    std::srand(123);
    const int n = 1000;
    ComplexMatrix A = ComplexMatrix::Random(n, n);
    A = (A + A.adjoint()).eval();
    ComplexVector b = ComplexVector::Random(n);
    const double shift = 1.0;

    run_test(A, b, shift);
}
