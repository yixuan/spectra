
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <random>  // Requires C++ 11

#include <Spectra/KrylovSchurGEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseRegularInverse.h>

#include "EigenIOFile.h"

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SpMatrix = Eigen::SparseMatrix<double>;

template <typename Solver>
void run_test(const SpMatrix& A, const SpMatrix& B, Solver& eigs, SortRule selection, bool allow_fail = false, bool invert_solution = false, double scale = 1)
{
    eigs.init();

    int nconv = eigs.compute(selection, 300, 1e-14);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    if (allow_fail && eigs.info() != CompInfo::Successful)
    {
        WARN("FAILED on this test");
        std::cout << "nconv = " << nconv << std::endl;
        std::cout << "niter = " << niter << std::endl;
        std::cout << "nops  = " << nops << std::endl;
        return;
    }
    else
    {
        INFO("nconv = " << nconv);
        INFO("niter = " << niter);
        INFO("nops  = " << nops);
        REQUIRE(eigs.info() == CompInfo::Successful);
    }

    Vector evals = eigs.eigenvalues() / scale;
    Matrix evecs = eigs.eigenvectors();

    Matrix resid = A.template selfadjointView<Eigen::Lower>() * evecs -
        B.template selfadjointView<Eigen::Lower>() * evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    if (invert_solution) {
        for (size_t i = 0; i < evals.size(); i++)
            evals[i] = 1 / evals[i];
    }

    for (size_t i = 0; i < evals.size(); i++)
        std::cout << "Eigenvalue #" << i << " = " << evals[i] << std::endl;

    // square root
    for (size_t i = 0; i < evals.size(); i++)
        evals[i] = sqrt(evals[i]);

    // output results
    for (size_t i = 0; i < evals.size(); i++)
        std::cout << "Eigen omega #" << i << " = " << evals[i] << std::endl;

    INFO("||AU - BUD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

void run_test_sets(SpMatrix& A, SpMatrix& B, int k, int m)
{
    using OpType = SparseSymMatProd<double>;
    using BOpType = SparseRegularInverse<double>;


    double scaleB = 1;
    scaleB = B.norm() / std::sqrt(B.cols());
    scaleB = std::pow(2, std::floor(std::log2(scaleB + 1)));

    B /= scaleB;

    OpType op(A);
    BOpType Bop(B, BOpType::SolverType::LU); // use SparseLU as solver
    KrylovSchurGEigsSolver<OpType, BOpType, GEigsMode::RegularInverse> eigs(op, Bop, k, m);
    //SymGEigsSolver<OpType, BOpType, GEigsMode::RegularInverse> eigs(op, Bop, k, m);

    bool invert_solution = true;

    if (invert_solution) {
        SECTION("Largest Magnitude")
        {
            run_test(A, B, eigs, SortRule::LargestMagn, true, invert_solution, scaleB);
        }
    }
    else {
        SECTION("Smallest Magnitude")
        {
            run_test(A, B, eigs, SortRule::SmallestMagn, true, invert_solution, scaleB);
        }
    }

}

TEST_CASE("Generalized eigensolver of sparse symmetric real matrix [16800x16800]", "[krylovschur]")
{
    SpMatrix A, B;
    Eigen::read_sparse("matrix_A.mtx", A, true);
    Eigen::read_sparse("matrix_B.mtx", B, true);
    int k = 6;
    int m = 2 * k >= 20 ? 2 * k : 20; // minimum subspace size 20

    run_test_sets(A, B, k, m);
}
