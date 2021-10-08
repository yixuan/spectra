#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <random>  // Requires C++ 11

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Spectra;

#include "catch.hpp"

using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
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

// Generate data for testing
Matrix gen_dense_data(int n)
{
    const Matrix mat = Eigen::MatrixXd::Random(n, n);
    return mat + mat.transpose();
}

SpMatrix gen_sparse_data(int n, double prob = 0.5)
{
    // Eigen solver only uses the lower triangle of mat,
    // so we don't need to make mat symmetric here.
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
void run_test(const MatType& mat, Solver& eigs, SortRule selection)
{
    eigs.init();
    int nconv = eigs.compute(selection);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    INFO("nops  = " << nops);
    REQUIRE(eigs.info() == CompInfo::Successful);

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    Matrix resid = mat.template selfadjointView<Eigen::Lower>() * evecs - evecs * evals.asDiagonal();
    const double err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

template <typename MatType>
void run_test_sets(const MatType& mat, int k, int m)
{
    constexpr bool is_dense = std::is_same<MatType, Matrix>::value;
    using DenseOp = DenseSymMatProd<double>;
    using SparseOp = SparseSymMatProd<double>;
    using OpType = typename std::conditional<is_dense, DenseOp, SparseOp>::type;
    using Scalar = typename OpType::Scalar;

    OpType op(mat);
    std::unique_ptr<LoggerBase<Scalar, Vector>> logger(new DerivedLogger<Scalar, Vector>());
    SymEigsSolver<OpType> eigs(op, k, m, std::move(logger));

    run_test(mat, eigs, SortRule::LargestMagn);
}

TEST_CASE("Eigensolver of symmetric real matrix [10x10]", "[eigs_sym]")
{
    std::srand(123);

    const Matrix A = gen_dense_data(10);
    int k = 3;
    int m = 6;

    run_test_sets(A, k, m);
}
