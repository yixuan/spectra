#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Spectra/JDSymEigsDPR.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


template<typename T>
using Matrix = Eigen::Matrix<T,-1,-1>;

template<typename T>
using Vector = Eigen::Matrix<T,-1,1>;

template<typename T>
using SpMatrix = Eigen::SparseMatrix<T>;

// Traits to obtain operation type from matrix type
template < template<typename> typename MatType, typename T>
struct OpTypeTrait
{
    using OpType = DenseSymMatProd<T>;
};

template <typename T>
struct OpTypeTrait<SpMatrix,T>
{
    using OpType = SparseSymMatProd<T>;
};

// Generate data for testing
template < typename T> 
Matrix<T> gen_sym_data_dense(int n)
{
    Matrix<T> mat = Matrix<T>::Random(n, n);
    Matrix<T> mat1 = mat + mat.transpose();
    mat1.diagonal() += Matrix<T>::Constant(n,1,n);
    return mat1;

}

template < typename T > 
SpMatrix<T> gen_sym_data_sparse(int n)
{
    // Eigen solver only uses the lower triangle of mat,
    // so we don't need to make mat symmetric here.
    double prob = 0.5;
    SpMatrix<T> mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = distr(gen) - 0.5;
            if (i == j) {
                mat.coeffRef(i,j) = n;
            }
        }
    }
    return mat;
}

template <template<typename> typename MatType,typename T>
void run_test(const MatType<T>& mat, int nev, SortRule selection)
{
    using OpType = typename OpTypeTrait<MatType, T>::OpType;
    OpType op(mat);
    JDSymEigsDPR< OpType> eigs(op,nev);
    //int nconv = eigs.compute(selection);

    int niter = eigs.num_iterations();
    //int nops = eigs.num_operations();

    //INFO("nconv = " << nconv);
    INFO("niter = " << niter);
    //INFO("nops  = " << nops);
    REQUIRE(eigs.info() == CompInfo::Successful);

    Vector<T> evals = eigs.eigenvalues();
    Matrix<T> evecs = eigs.eigenvectors();

    Matrix<T> resid = mat.template selfadjointView<Eigen::Lower>() * evecs - evecs * evals.asDiagonal();
    const T err = resid.array().abs().maxCoeff();

    INFO("||AU - UD||_inf = " << err);
    REQUIRE(err == Approx(0.0).margin(1e-9));
}

template< template<typename> typename MatType, typename T>
void run_test_set(const MatType<T>& mat, int k)
{
    SECTION("Largest Magnitude")
    {
        run_test<MatType,T>(mat, k, SortRule::LargestMagn);
    }
    SECTION("Largest Value")
    {
        run_test<MatType,T>(mat, k, SortRule::LargestAlge);
    }
    SECTION("Smallest Magnitude")
    {
        run_test<MatType,T>(mat, k, SortRule::SmallestMagn);
    }
    SECTION("Smallest Value")
    {
        run_test<MatType,T>(mat, k, SortRule::SmallestAlge);
    }
    SECTION("Both Ends")
    {
        run_test<MatType,T>(mat, k, SortRule::BothEnds);
    }
}
TEMPLATE_TEST_CASE("Davidson Solver of dense symmetric real matrix [10x10]","", float, double)
{
    std::srand(123);
    const Matrix<TestType> A = gen_sym_data_dense<TestType>(10);
    int k = 3;
    run_test_set<Matrix,TestType>(A,k);
}

TEMPLATE_TEST_CASE("Davidson Solver of dense symmetric real matrix [100x100]","", float, double)
{
    std::srand(123);
    const Matrix<TestType> A = gen_sym_data_dense<TestType>(100);
    int k = 10;
    run_test_set<Matrix,TestType>(A,k);
}

TEMPLATE_TEST_CASE("Davidson Solver of sparse symmetric real matrix [10x10]","", float, double)
{
    std::srand(123);
    int k = 3;
    const SpMatrix<TestType> A = gen_sym_data_sparse<TestType>(10);
    run_test_set<SpMatrix,TestType>(A,k);
}

TEMPLATE_TEST_CASE("Davidson Solver of sparse symmetric real matrix [100x100]","", float, double)
{
    std::srand(123);
    int k = 10;
    const SpMatrix<TestType> A = gen_sym_data_sparse<TestType>(100);
    run_test_set<SpMatrix,TestType>(A,k);
}
