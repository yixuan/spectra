#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>

#include <Spectra/Util/SearchSpace.h>


using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<double>;
using Index = Eigen::Index;


TEST_CASE("update_operator_basis_product", "[SearchSpace]")
{
    SearchSpace<double> space;

}

TEST_CASE("full_update", "[SearchSpace]")
{
  
}

TEST_CASE("restart", "[SearchSpace]")
{
   
}

TEST_CASE("append_new_vectors_to_basis", "[SearchSpace]")
{
   
}