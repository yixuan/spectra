#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/contrib/LOBPCGSolver.h>
#include <cmath>

#include "catch.hpp"

using Scalar = long double;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using SpMatrix = Eigen::SparseMatrix<Scalar>;

void run_lobpcg_test(bool generalized, bool iterative)
{
    const int n = 12;
    const int nev = 2;
    Matrix A = Matrix::Zero(n, n);
    Matrix B = Matrix::Identity(n, n);
    Matrix initial = Matrix::Zero(n, nev);
    Matrix preconditioner = Matrix::Zero(n, n);
    for (int i = 0; i < n; i++)
    {
        if (generalized)
            B(i, i) = 1 + Scalar(0.25) * i;
        A(i, i) = (i + 1) * B(i, i);
        preconditioner(i, i) = Scalar(1) / (i + 1);
        if (iterative)
        {
            initial(i, 0) = std::sin((i + 1) * Scalar(0.731)) + Scalar(0.2);
            initial(i, 1) = std::cos((i + 1) * Scalar(0.457)) - Scalar(0.1);
        }
    }
    if (!iterative)
    {
        initial(0, 0) = 2;
        initial(1, 0) = -1;
        initial(0, 1) = 1;
        initial(1, 1) = 3;
    }

    SpMatrix sparse_A = A.sparseView();
    SpMatrix sparse_B = B.sparseView();
    SpMatrix sparse_initial = initial.sparseView();
    SpMatrix sparse_preconditioner = preconditioner.sparseView();
    Spectra::LOBPCGSolver<> solver(sparse_A, sparse_initial);
    if (generalized)
        solver.setB(sparse_B);
    if (iterative)
        solver.setPreconditioner(sparse_preconditioner);
    REQUIRE(solver.eigenvectors().size() == 0);
    const Scalar tolerance = iterative ? Scalar(1e-7) : Scalar(1e-10);
    solver.compute(100, tolerance);
    REQUIRE(solver.info() == Eigen::Success);

    Vector evals = solver.eigenvalues();
    Matrix evecs = solver.eigenvectors();
    Matrix reported = solver.residuals();
    REQUIRE(evals.size() == nev);
    REQUIRE(evecs.rows() == n);
    REQUIRE(evecs.cols() == nev);
    REQUIRE(reported.rows() == n);
    REQUIRE(reported.cols() == nev);
    REQUIRE(evals.allFinite());
    REQUIRE(evecs.allFinite());
    REQUIRE(reported.allFinite());

    Matrix residual = A * evecs - B * evecs * evals.asDiagonal();
    REQUIRE(residual.cwiseAbs().maxCoeff() <= tolerance * n);
    REQUIRE((residual - reported).cwiseAbs().maxCoeff() < Scalar(1e-10));
    Matrix orthogonality = evecs.transpose() * B * evecs - Matrix::Identity(nev, nev);
    REQUIRE(orthogonality.cwiseAbs().maxCoeff() < Scalar(1e-8));
    REQUIRE(evals(0) == Approx(1).margin(1e-8));
    REQUIRE(evals(1) == Approx(2).margin(1e-8));
}

TEST_CASE("LOBPCG eigenvectors use the original coordinates", "[lobpcg]")
{
    SECTION("Ordinary problem converged in the initial subspace")
    {
        run_lobpcg_test(false, false);
    }
    SECTION("Generalized problem converged in the initial subspace")
    {
        run_lobpcg_test(true, false);
    }
    SECTION("Iterative problem with a preconditioner")
    {
        run_lobpcg_test(false, true);
    }
}
