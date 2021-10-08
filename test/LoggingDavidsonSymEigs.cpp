#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <iostream>
#include <iomanip>
#include <type_traits>
#include <Spectra/DavidsonSymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Spectra;

#include "catch.hpp"

using Index = Eigen::Index;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using SpMatrix = Eigen::SparseMatrix<T>;

using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Traits to obtain operation type from matrix type
template <typename MatType>
struct OpTypeTrait
{
    using Scalar = typename MatType::Scalar;
    using OpType = DenseSymMatProd<Scalar>;
};
template <typename T>
struct OpTypeTrait<SpMatrix<T>>
{
    using OpType = SparseSymMatProd<T>;
};

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
template <typename Matrix>
Matrix gen_sym_data_dense(int n)
{
    Matrix mat = 0.03 * Matrix::Random(n, n);
    Matrix mat1 = mat + mat.transpose();
    for (Eigen::Index i = 0; i < n; i++)
    {
        mat1(i, i) += i + 1;
    }
    return mat1;
}

template <typename SpMatrix>
SpMatrix gen_sym_data_sparse(int n)
{
    // Eigen solver only uses the lower triangle of mat,
    // so we don't need to make mat symmetric here.
    double prob = 0.5;
    SpMatrix mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = 0.1 * (distr(gen) - 0.5);
            if (i == j)
            {
                mat.coeffRef(i, j) = i + 1;
            }
        }
    }
    return mat;
}

template <typename MatType>
void run_test(const MatType& mat, int nev, SortRule selection)
{
    using OpType = typename OpTypeTrait<MatType>::OpType;
    using Scalar = typename OpType::Scalar;
    OpType op(mat);
    std::unique_ptr<LoggerBase<Scalar, Vector<Scalar>>> logger(new DerivedLogger<Scalar, Vector<Scalar>>());
    DavidsonSymEigsSolver<OpType> eigs(op, nev, std::move(logger));
    int nconv = eigs.compute(selection);

    int niter = eigs.num_iterations();
    REQUIRE(nconv == nev);
    INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    REQUIRE(eigs.info() == CompInfo::Successful);
    using T = typename OpType::Scalar;
    Vector<T> evals = eigs.eigenvalues();
    Matrix<T> evecs = eigs.eigenvectors();

    Matrix<T> resid = op * evecs - evecs * evals.asDiagonal();
    const T err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err < 100 * Eigen::NumTraits<T>::dummy_precision());
}

template <typename MatType>
void run_test_set(const MatType& mat, int k)
{
    run_test<MatType>(mat, k, SortRule::LargestAlge);
}

TEMPLATE_TEST_CASE("Davidson Solver of sparse symmetric real matrix [1000x1000]", "", double)
{
    std::srand(123);
    int k = 10;
    const SpMatrix<TestType> A = gen_sym_data_sparse<SpMatrix<TestType>>(1000);
    run_test_set<SpMatrix<TestType>>(A, k);
}
