
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Eigen/Dense>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::Index;

TEST_CASE("matrix operations", "[DenseGenMatProd]")
{
    std::srand(123);
    constexpr Index n = 100;

    MatrixXd mat1 = MatrixXd::Random(n, n);
    MatrixXd mat2 = MatrixXd::Random(n, n);

    DenseGenMatProd<double> dense1(mat1);
    MatrixXd xs = dense1 * mat2;
    MatrixXd ys = mat1 * mat2;

    INFO("The matrix-matrix product must be the same as in eigen.")
    REQUIRE(xs.isApprox(ys));
    INFO("The accesor operator must produce the same element as in eigen")
    REQUIRE(mat1(23, 87) == dense1(23, 87));
}
