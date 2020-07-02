#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>

#include <Spectra/JDSymEigsBase.h>
#include <Spectra/MatOp/DenseGenMatProd.h>

#include <Spectra/Util/RitzPairs.h>
#include <Spectra/Util/SelectionRule.h>
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

template <typename OpType>
class JDMock : public JDSymEigsBase<OpType>
{
public:
    JDMock(OpType& op, Index nev) :
        JDSymEigsBase<OpType>(op, nev) {}
    Matrix SetupInitialSearchSpace(SortRule) const
    {
        Matrix V = Matrix::Random(10, 10);
        QR_orthogonalisation(V);
        return V;
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
    JDMock<DenseGenMatProd<double>> eigs(op, 5);

    REQUIRE(eigs.num_iterations() == 0);
    REQUIRE(eigs.info() == Spectra::CompInfo::NotComputed);
}

TEST_CASE("Sorting", "[RitzPairs]")
{
    const Matrix A = Eigen::MatrixXd::Random(10, 10);
    DenseGenMatProd<double> op(A);
    JDMock<DenseGenMatProd<double>> eigs(op, 5);
    eigs.search_space_.BasisVectors() = eigs.SetupInitialSearchSpace();
    eigs.search_space_.update_operator_basis_product(eigs.matrix_operator_);
    eigs.ritz_pairs_.compute_eigen_pairs(eigs.search_space_);

    // pair.values_ = Eigen::Vector::Random(10);
    // pair.vectors_ = Eigen::Matrix::Random(10, 10);
    // pair.sort(Spectra::LargestReal)
}

TEST_CASE("Convergence", "[RitzPairs]")
{
}

TEST_CASE("compute_eigen_pairs", "[RitzPairs]")
{
}