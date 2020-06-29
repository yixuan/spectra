#include <Eigen/Core>
#include <Spectra/LinAlg/GramSchmidt.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;


TEST_CASE("complete orthonormalization", "[Gram-Schmidt]")
{
    std::srand(123);
    const int n = 100;
    const double tol = 1e-12;

    MatrixXd mat = MatrixXd::Random(n, n);
    Gramschmidt<double> gs{mat};
    MatrixXd basis = gs.orthonormalize();
    MatrixXd xs = basis.transpose() * basis;
     INFO("The orthonormalized basis must fulfill that basis.T * basis = I");
    REQUIRE(xs.isIdentity(tol));
}

TEST_CASE("Partial orthonormalization", "[Gram-Schmidt]")
{
    std::srand(123);
    const int n = 100;
    const double tol = 1e-12;

    MatrixXd mat = MatrixXd::Random(n, n);
    Gramschmidt<double> gs{mat};
    MatrixXd basis = gs.orthonormalize();
    MatrixXd xs = basis.transpose() * basis;
     INFO("The orthonormalized basis must fulfill that basis.T * basis = I");
    REQUIRE(xs.isIdentity(tol));
}
