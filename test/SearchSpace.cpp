#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/Util/SearchSpace.h>
#include <Spectra/LinAlg/Orthogonalization.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<double>;
using Index = Eigen::Index;

TEST_CASE("CompleteSearchSpace", "[SearchSpace]")
{
    SearchSpace<double> space;
    Matrix initial_space = Matrix::Random(10, 3);
    Spectra::twice_is_enough_orthogonalisation(initial_space);
    space.InitializeSearchSpace(initial_space);
    REQUIRE(space.BasisVectors().cols() == 3);
    REQUIRE(space.OperatorBasisProduct().cols() == 0);

    Matrix A = Eigen::MatrixXd::Random(10, 10);
    Matrix B = A + A.transpose();
    DenseGenMatProd<double> op(B);

    space.update_operator_basis_product(op);
    REQUIRE(space.BasisVectors().cols() == 3);
    REQUIRE(space.OperatorBasisProduct().cols() == 3);
    REQUIRE(space.OperatorBasisProduct().isApprox(B * initial_space));

    Matrix append_space = Matrix::Random(10, 3);
    space.extend_basis(append_space);
    REQUIRE((space.BasisVectors().transpose() * space.BasisVectors()).isIdentity(1e-12));
    REQUIRE(space.BasisVectors().cols() == 6);
    REQUIRE(space.OperatorBasisProduct().cols() == 3);
    space.update_operator_basis_product(op);
    REQUIRE(space.OperatorBasisProduct().cols() == 6);

    RitzPairs<double> ritzpair;
    ritzpair.compute_eigen_pairs(space);
    REQUIRE(ritzpair.size() == 6);
    space.restart(ritzpair, 2);

    REQUIRE(space.BasisVectors().cols() == 2);
    REQUIRE(space.OperatorBasisProduct().cols() == 2);
}
