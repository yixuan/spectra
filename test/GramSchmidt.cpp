#include <Eigen/Core>
#include <Spectra/LinAlg/GramSchmidt.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;

template <typename Matrix>
void check_orthogonality(const Matrix& basis)
{
    const double tol = 1e-12;
    Matrix xs = basis.transpose() * basis;
    INFO("The orthonormalized basis must fulfill that basis.T * basis = I");
    REQUIRE(xs.isIdentity(tol));
}

TEST_CASE("complete orthonormalization", "[Gram-Schmidt]")
{
    std::srand(123);
    const Index n = 100;

    MatrixXd mat = MatrixXd::Random(n, n);
    Gramschmidt<double> gs{mat};
    check_orthogonality(gs.orthonormalize());
}

TEST_CASE("Partial orthonormalization", "[Gram-Schmidt]")
{
    std::srand(123);
    const Index n = 100;

    // Create a n x 20 orthonormal basis
    MatrixXd mat = MatrixXd::Random(n, n - 20);
    Gramschmidt<double> gs{mat};
    mat.leftCols(n - 20) = gs.orthonormalize();

    mat.conservativeResize(Eigen::NoChange, n);
    mat.rightCols(20) = MatrixXd::Random(n, 20);

    // Orthogonalize from 80 onwards
    Gramschmidt<double> new_gs{mat, 80};
    check_orthogonality(new_gs.orthonormalize());
}