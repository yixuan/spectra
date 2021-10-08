#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <random>  // Requires C++ 11
#include <memory>  // Requires C++ 11

#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/LoggerBase.h>

using namespace Spectra;

#include "catch.hpp"

using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<double>;
using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Derived Logger
template <typename Scalar, typename Vector>
class DerivedLogger : public LoggerBase<Scalar, Vector>
{
    // This derived logging class could have some reference to an ostream or call to another class that wraps ostreams etc.
public:
    DerivedLogger(){};
    void iteration_log(const IterationData<Scalar, Vector>& data) override
    {
        std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "    Iteration                       :   " << data.iteration << std::endl;
        std::cout << "    Number of converged eigenvalues :   " << data.number_of_converged << std::endl;
        std::cout << "    Size of subspace                :   " << data.subspace_size << std::endl;
        std::cout << "    ------------------------------------------------------------------------              " << std::endl;
        REQUIRE(data.residues.size() == data.current_eigenvalues.size());
        REQUIRE(data.residues.size() == data.current_eig_converged.size());

        std::cout << "       " << std::setw(20) << "Current Eigenvalue" << std::setw(20) << "Converged?" << std::setw(20) << "Residue" << std::endl;
        std::cout << "    ------------------------------------------------------------------------              " << std::endl;
        for (int i = 0; i < data.current_eigenvalues.size(); i++)
        {
            std::cout << "       " << std::setw(20) << data.current_eigenvalues[i] << std::setw(20) << data.current_eig_converged[i] << std::setw(20) << data.residues[i] << std::endl;
        }
        std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
    }
};

// Generate random sparse matrix
SpMatrix gen_sparse_data(int n, double prob = 0.5)
{
    SpMatrix mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = distr(gen) - 0.5;
        }
    }
    return mat;
}

template <typename MatType, typename Solver>
void run_test(const MatType& mat, Solver& eigs, SortRule selection, bool allow_fail = false)
{
    eigs.init();
    // maxit = 300 to reduce running time for failed cases
    int nconv = eigs.compute(selection, 300);
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

    ComplexVector evals = eigs.eigenvalues();
    ComplexMatrix evecs = eigs.eigenvectors();

    ComplexMatrix resid = mat * evecs - evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

template <typename MatType>
void run_test_sets(const MatType& A, int k, int m)
{
    using OpType = DenseGenMatProd<double>;
    using Scalar = typename OpType::Scalar;

    OpType op(A);
    std::unique_ptr<LoggerBase<Scalar, ComplexVector>> logger(new DerivedLogger<Scalar, ComplexVector>());

    GenEigsSolver<OpType> eigs(op, k, m, std::move(logger));
    run_test(A, eigs, SortRule::LargestMagn);
}

TEST_CASE("Eigensolver of general real matrix [10x10]", "[eigs_gen]")
{
    std::srand(123);

    const Matrix A = Matrix::Random(10, 10);
    int k = 3;
    int m = 6;

    run_test_sets(A, k, m);
}
