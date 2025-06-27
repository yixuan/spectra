// Test ../include/Spectra/LinAlg/UpperHessenbergEigen.h and
//      ../include/Spectra/LinAlg/TridiagEigen.h
#include <chrono>
#include <iostream>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Spectra/LinAlg/UpperHessenbergEigen.h>
#include <Spectra/LinAlg/TridiagEigen.h>

#include "catch.hpp"

using namespace Spectra;

// https://stackoverflow.com/a/34781413
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Complex = std::complex<double>;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;

TEST_CASE("Eigen decomposition of real upper Hessenberg matrix", "[Eigen]")
{
    std::srand(123);
    const int n = 100;
    Matrix M = Matrix::Random(n, n);
    // H is upper Hessenberg
    Matrix H = M.triangularView<Eigen::Upper>();
    H.diagonal(-1) = M.diagonal(-1);

    UpperHessenbergEigen<double> decomp(H);
    ComplexVector evals = decomp.eigenvalues();
    ComplexMatrix evecs = decomp.eigenvectors();

    // Test accuracy
    ComplexMatrix err = H * evecs - evecs * evals.asDiagonal();
    INFO("||HU - UD||_inf = " << err.cwiseAbs().maxCoeff());
    REQUIRE(err.cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-12));

    TimePoint t1, t2;
    t1 = Clock::now();
    for (int i = 0; i < 100; i++)
    {
        UpperHessenbergEigen<double> decomp(H);
        ComplexVector evals = decomp.eigenvalues();
        ComplexMatrix evecs = decomp.eigenvectors();
    }
    t2 = Clock::now();
    std::cout << "Elapsed time for UpperHessenbergEigen: "
              << (t2 - t1).count() << " ms\n";

    t1 = Clock::now();
    for (int i = 0; i < 100; i++)
    {
        Eigen::EigenSolver<Matrix> decomp(H);
        ComplexVector evals = decomp.eigenvalues();
        ComplexMatrix evecs = decomp.eigenvectors();
    }
    t2 = Clock::now();
    std::cout << "Elapsed time for Eigen::EigenSolver: "
              << (t2 - t1).count() << " ms\n";
}

TEST_CASE("Eigen decomposition of real symmetric tridiagonal matrix", "[Eigen]")
{
    std::srand(123);
    const int n = 100;
    Matrix M = Matrix::Random(n, n);
    // H is symmetric tridiagonal
    Matrix H = Matrix::Zero(n, n);
    H.diagonal() = M.diagonal();
    H.diagonal(-1) = M.diagonal(-1);
    H.diagonal(1) = M.diagonal(-1);

    TridiagEigen<double> decomp(H);
    Vector evals = decomp.eigenvalues();
    Matrix evecs = decomp.eigenvectors();

    // Test accuracy
    Matrix err = H * evecs - evecs * evals.asDiagonal();
    INFO("||HU - UD||_inf = " << err.cwiseAbs().maxCoeff());
    REQUIRE(err.cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-12));

    TimePoint t1, t2;
    t1 = Clock::now();
    for (int i = 0; i < 100; i++)
    {
        TridiagEigen<double> decomp(H);
        Vector evals = decomp.eigenvalues();
        Matrix evecs = decomp.eigenvectors();
    }
    t2 = Clock::now();
    std::cout << "Elapsed time for TridiagEigen: "
              << (t2 - t1).count() << " ms\n";

    t1 = Clock::now();
    for (int i = 0; i < 100; i++)
    {
        Eigen::SelfAdjointEigenSolver<Matrix> decomp(H);
        Vector evals = decomp.eigenvalues();
        Matrix evecs = decomp.eigenvectors();
    }
    t2 = Clock::now();
    std::cout << "Elapsed time for Eigen::SelfAdjointEigenSolver: "
              << (t2 - t1).count() << " ms\n";
}

TEST_CASE("Eigen decomposition of complex upper Hessenberg matrix", "[Eigen]")
{
    std::srand(123);
    const int n = 100;
    ComplexMatrix M = ComplexMatrix::Random(n, n);
    // H is upper Hessenberg
    ComplexMatrix H = M.triangularView<Eigen::Upper>();
    H.diagonal(-1) = M.diagonal(-1);

    UpperHessenbergEigen<Complex> decomp(H);
    ComplexVector evals = decomp.eigenvalues();
    ComplexMatrix evecs = decomp.eigenvectors();

    // Test accuracy
    ComplexMatrix err = H * evecs - evecs * evals.asDiagonal();
    INFO("||HU - UD||_inf = " << err.cwiseAbs().maxCoeff());
    REQUIRE(err.cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-12));

    TimePoint t1, t2;
    t1 = Clock::now();
    for (int i = 0; i < 100; i++)
    {
        UpperHessenbergEigen<Complex> decomp(H);
        ComplexVector evals = decomp.eigenvalues();
        ComplexMatrix evecs = decomp.eigenvectors();
    }
    t2 = Clock::now();
    std::cout << "Elapsed time for UpperHessenbergEigen: "
              << (t2 - t1).count() << " ms\n";

    t1 = Clock::now();
    for (int i = 0; i < 100; i++)
    {
        Eigen::ComplexEigenSolver<ComplexMatrix> decomp(H);
        ComplexVector evals = decomp.eigenvalues();
        ComplexMatrix evecs = decomp.eigenvectors();
    }
    t2 = Clock::now();
    std::cout << "Elapsed time for Eigen::EigenSolver: "
              << (t2 - t1).count() << " ms\n";
}
