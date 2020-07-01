#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Spectra/JDSymEigsDPR.h>
#include <Spectra/MatOp/DenseGenMatProd.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEMPLATE_TEST_CASE("Constructing JDSymEigsDPR", "[JDSymEigsDPR]", float, double)
{
    using Matrix = Eigen::Matrix<TestType, Eigen::Dynamic, Eigen::Dynamic>;
    const Matrix A = Matrix::Random(10, 10);
    DenseGenMatProd<TestType> op(A);
    JDSymEigsDPR<TestType, DenseGenMatProd<TestType>> eigs{op, 5};
    REQUIRE(eigs.num_iterations() == 0);
    REQUIRE(eigs.info() == Spectra::CompInfo::NotComputed);
}