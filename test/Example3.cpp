// Example reported in Issue #115
// https://github.com/yixuan/spectra/issues/115
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
using TriVec = std::vector<Eigen::Triplet<double>>;
using OpType = SparseSymMatProd<double>;
using BOpType = SparseCholesky<double>;

// (Highest) Eigenvalues of: ([A] + lam [B])x = 0
//		A := M,				(pos. semi definite)
//		B := [C + shift*M]	(pos. definite)
//		-> f = sqrt(1/lam - s)/2pi
void run_test(SpMat& M, SpMat& C, double shift, int nef, int ncv)
{
    SpMat A = M;
    SpMat B = C + shift * M;

    INFO("A =\n " << Matrix(A));
    INFO("B =\n " << Matrix(B));

    OpType op(A);
    BOpType Bop(B);
    REQUIRE(Bop.info() == CompInfo::Successful);

    SymGEigsSolver<OpType, BOpType, GEigsMode::Cholesky> eigs(op, Bop, nef, ncv);
    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    INFO("nops  = " << nops);
    REQUIRE(eigs.info() == CompInfo::Successful);

    Vector evals = eigs.eigenvalues();
    Vector lam = 1 / (eigs.eigenvalues().array()) - shift;
    Matrix evecs = eigs.eigenvectors();

    Matrix resid = A.template selfadjointView<Eigen::Lower>() * evecs -
        B.template selfadjointView<Eigen::Lower>() * evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    INFO("evals = " << evals.transpose());
    INFO("lam = " << lam.transpose());
    INFO("U'BU =\n " << evecs.transpose() * B * evecs);
    INFO("||AU - BUD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

TEST_CASE("Example #115, case 1", "[example_115]")
{
    TriVec C_tri = {
        {0, 0, 1.1807575e+08},
        {1, 1, 304744.5},
        {1, 5, -152372.25},
        {2, 2, 304744.5},
        {2, 4, 152372.25},
        {3, 3, 15403.85},
        {4, 2, 152372.25},
        {4, 4, 101581.5},
        {5, 1, -152372.25},
        {5, 5, 101581.5},
    };

    TriVec M_tri = {
        {0, 0, 1000.0},
        {1, 1, 1000.0},
        {2, 2, 1000.0},
    };

    SpMat C(6, 6);
    C.setFromTriplets(C_tri.begin(), C_tri.end());

    SpMat M(6, 6);
    M.setFromTriplets(M_tri.begin(), M_tri.end());

    const double shift = 1.0e5;

    run_test(M, C, shift, 4, 5);
}

TEST_CASE("Example #115, case 2", "[example_115]")
{
    TriVec C_tri = {
        {0, 0, 2.361515e+08},
        {1, 1, 609489.01},
        {2, 2, 609489.01},
        {3, 3, 30807.7},
        {4, 4, 203163},
        {5, 5, 203163},
        {6, 0, -1.1807575e+08},
        {0, 6, -1.1807575e+08},
        {6, 6, 1.1807575e+08},
        {7, 1, -304744.5},
        {1, 7, -304744.5},
        {7, 5, -152372.25},
        {5, 7, -152372.25},
        {7, 7, 304744.5},
        {8, 2, -304744.5},
        {2, 8, -304744.5},
        {8, 4, 152372.25},
        {4, 8, 152372.25},
        {8, 8, 304744.5},
        {9, 3, -15403.85},
        {3, 9, -15403.85},
        {9, 9, 15403.85},
        {10, 2, -152372.25},
        {2, 10, -152372.25},
        {10, 4, 50790.751},
        {4, 10, 50790.751},
        {10, 8, 152372.25},
        {8, 10, 152372.25},
        {10, 10, 101581.5},
        {11, 1, 152372.25},
        {1, 11, 152372.25},
        {11, 5, 50790.751},
        {5, 11, 50790.751},
        {11, 7, -152372.25},
        {7, 11, -152372.25},
        {11, 11, 101581.5},
    };

    TriVec M_tri = {
        {0, 0, 1000.0},
        {1, 1, 1000.0},
        {2, 2, 1000.0},
    };

    SpMat C(12, 12);
    C.setFromTriplets(C_tri.begin(), C_tri.end());

    SpMat M(12, 12);
    M.setFromTriplets(M_tri.begin(), M_tri.end());

    const double shift = 1.0e5;

    run_test(M, C, shift, 5, 8);
}
