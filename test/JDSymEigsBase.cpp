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
using Index = Eigen::Index;
template <typename Scalar,
          typename OpType>
class JDMock : public JDSymEigsBase<Scalar, OpType>
{
public:
    JDMock(OpType& op, Index nev) :
        JDSymEigsBase<Scalar, OpType>(op, nev) {}
    Matrix SetupInitialSearchSpace(SortRule selection) const
    {
        return Matrix::Zero(0, 0);
    }

    Matrix CalculateCorrectionVector() const
    {
        return Matrix::Zero(0, 0);
    }
};

TEST_CASE("Constructing JDSymObject", "[eigs_gen]")
{
    const Matrix A = Eigen::MatrixXd::Random(10, 10);
    DenseGenMatProd<double> op(A);
    JDMock<double, DenseGenMatProd<double>> eigs(op, 5);

    REQUIRE(eigs.num_iterations() == 0);
    REQUIRE(eigs.info() == Spectra::CompInfo::NotComputed);
}