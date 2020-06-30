#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>

#include <Spectra/Util/RitzPairs.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<double>;
using Index = Eigen::Index;

TEST_CASE("Sorting", "[RitzPairs]")
{
    RitzPairs<double> pair;
}

TEST_CASE("Convergence", "[RitzPairs]")
{
}

TEST_CASE("compute_eigen_pairs", "[RitzPairs]")
{
}