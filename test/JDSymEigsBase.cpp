#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <random>  // Requires C++ 11

#include <Spectra/JDSymEigsBase.h>
#include <Spectra/MatOp/DenseGenMatProd.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<double>;

TEST_CASE("Constructing JDSymObject", "[eigs_gen]")
{
    const Matrix A = Eigen::MatrixXd::Random(10, 10);
    DenseGenMatProd<double> op(A);
    JDSymEigsBase<double, DenseGenMatProd<double> > eigs(op, 5);
}