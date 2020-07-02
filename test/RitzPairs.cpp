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
        Matrix V = Matrix::Random(10, 5);
        QR_orthogonalisation(V);
        return V;
    }

    Matrix CalculateCorrectionVector() const { return Matrix::Zero(0, 0); }

    void InitSearchSpace()
    {
        Matrix V = this->SetupInitialSearchSpace(Spectra::SortRule::LargestAlge);
        this->search_space_.BasisVectors() = V;
        this->search_space_.OperatorBasisProduct() = this->matrix_operator_ * V;
    }

    void InitResidues() { this->ritz_pairs_.Residues() = 1E-6 * Matrix::Random(10, 10); }
    void InitRitzPairs() { this->ritz_pairs_.compute_eigen_pairs(this->search_space_); }
    void SortRitzPairs() { this->ritz_pairs_.sort(Spectra::SortRule::LargestAlge); }
    bool checkConvergence(double tol, Index nev) const { return this->ritz_pairs_.check_convergence(tol, nev); }
};

TEST_CASE("compute_eigen_pairs", "[RitzPairs]")
{
    Matrix A = Eigen::MatrixXd::Random(10, 10);
    Matrix B = A + A.transpose();
    DenseGenMatProd<double> op(B);
    JDMock<DenseGenMatProd<double>> eigs(op, 5);

    eigs.InitSearchSpace();

    eigs.InitRitzPairs();
    eigs.SortRitzPairs();
    Vector ev = eigs.eigenvalues();

    for (Index i = 1; i < 5; i++)
    {
        CHECK(ev(i) < ev(i - 1));
    }
}

TEST_CASE("Convergence", "[RitzPairs]")
{
    Matrix A = Eigen::MatrixXd::Random(10, 10);
    Matrix B = A + A.transpose();
    DenseGenMatProd<double> op(B);
    JDMock<DenseGenMatProd<double>> eigs(op, 5);
    eigs.InitResidues();
    // CHECK(eigs.checkConvergence(1E-3, 5));
}
