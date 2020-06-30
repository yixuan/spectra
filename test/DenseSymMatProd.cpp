
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <complex>

using Eigen::Matrix;
using Eigen::Index;

TEMPLATE_TEST_CASE("matrix operations", "[DenseSymMatProd]", float, double)
{
    std::srand(123);
    constexpr Index n = 100;

    Matrix<TestType, -1, -1> mat = Matrix<TestType, -1, -1>::Random(n, n);
    Matrix<TestType, -1, -1> mat1 = mat + mat.transpose();  // It needs to be symetric
    Matrix<TestType, -1, -1> mat2 = Matrix<TestType, -1, -1>::Random(n, n);

    DenseSymMatProd<TestType> dense1(mat1);
    Matrix<TestType, -1, -1> xs = dense1 * mat2;
    Matrix<TestType, -1, -1> ys = mat1 * mat2;

    INFO("The matrix-matrix product must be the same as in eigen.")
    REQUIRE(xs.isApprox(ys));
    INFO("The accesor operator must produce the same element as in eigen")
    REQUIRE(mat1(15, 23) == dense1(23, 15));
}
