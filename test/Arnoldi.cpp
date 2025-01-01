// Test ../include/Spectra/LinAlg/Arnoldi.h and
//      ../include/Spectra/LinAlg/Lanczos.h
#include <Eigen/Core>
#include <Spectra/LinAlg/Arnoldi.h>
#include <Spectra/LinAlg/Lanczos.h>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseHermMatProd.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;

// clang-format off
template <typename FacType, typename MatType>
void run_test(FacType& fac, const MatType& A, int m)
{
    using VecType = Eigen::Matrix<typename MatType::Scalar, Eigen::Dynamic, 1>;

    // Initialization
    const int n = A.rows();
    VecType v0 = VecType::Random(n);
    Eigen::Map<const VecType> v0map(v0.data(), n);
    Index op_counter = 0;
    fac.init(v0map, op_counter);

    // After initialization, the subspace dimension is one
    const int k1 = fac.subspace_dim();
    INFO("k1 = " << k1);
    REQUIRE(k1 == 1);

    // A*V = V*H + f*e'
    constexpr double tol = 1e-12;
    MatType V = fac.matrix_V().leftCols(k1);
    MatType H = fac.matrix_H().topLeftCorner(k1, k1);
    MatType resid = A * V - V * H;
    VecType f = fac.vector_f();
    INFO("AV - VH = \n" << resid);
    INFO("f = " << f.transpose());
    // Test residual
    REQUIRE((resid.template rightCols<1>() - f).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Factorization to m/2
    fac.factorize_from(1, m / 2, op_counter);
    const int k2 = fac.subspace_dim();
    INFO("k2 = " << k2);
    REQUIRE(k2 == m / 2);

    V = fac.matrix_V().leftCols(k2);
    H = fac.matrix_H().topLeftCorner(k2, k2);
    resid = A * V - V * H;
    f = fac.vector_f();
    INFO("AV - VH = \n" << resid);
    INFO("f = " << f.transpose());
    // Test residual
    REQUIRE(resid.leftCols(k2 - 1).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
    REQUIRE((resid.template rightCols<1>() - f).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Factorization to m
    fac.factorize_from(m / 2, m, op_counter);
    const int k3 = fac.subspace_dim();
    INFO("k3 = " << k3);
    REQUIRE(k3 == m);

    V = fac.matrix_V().leftCols(k3);
    H = fac.matrix_H().topLeftCorner(k3, k3);
    resid = A * V - V * H;
    f = fac.vector_f();
    MatType VHV = V.adjoint() * V;
    INFO("V = \n" << V);
    INFO("H = \n" << H);
    INFO("V^H * V = \n" << VHV);
    INFO("AV - VH = \n" << resid);
    INFO("f = " << f.transpose());
    // Test orthogonality of V
    MatType iden = MatType::Identity(m, m);
    REQUIRE((VHV - iden).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
    // Test residual
    REQUIRE(resid.leftCols(k3 - 1).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
    REQUIRE((resid.template rightCols<1>() - f).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
}
// clang-format on

TEST_CASE("Arnoldi factorization of general real matrix", "[Arnoldi]")
{
    std::srand(123);
    const int n = 10;
    Matrix A = Matrix::Random(n, n);
    const int m = 6;

    using AOpType = DenseGenMatProd<double>;
    using ArnoldiOpType = ArnoldiOp<double, AOpType, IdentityBOp>;

    AOpType Aop(A);
    ArnoldiOpType arn(Aop, IdentityBOp());
    Arnoldi<double, ArnoldiOpType> fac(arn, m);

    run_test(fac, A, m);
}

TEST_CASE("Lanczos factorization of symmetric real matrix", "[Lanczos]")
{
    std::srand(123);
    const int n = 10;
    Matrix A = Matrix::Random(n, n);
    A = (A + A.transpose()).eval();
    const int m = 6;

    using AOpType = DenseSymMatProd<double>;
    using ArnoldiOpType = ArnoldiOp<double, AOpType, IdentityBOp>;

    AOpType Aop(A);
    ArnoldiOpType arn(Aop, IdentityBOp());
    Lanczos<double, ArnoldiOpType> fac(arn, m);

    run_test(fac, A, m);
}

TEST_CASE("Arnoldi factorization of general complex matrix", "[Arnoldi]")
{
    std::srand(123);
    const int n = 10;
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(n, n);
    const int m = 6;

    using Scalar = std::complex<double>;
    using AOpType = DenseGenMatProd<Scalar>;
    using ArnoldiOpType = ArnoldiOp<Scalar, AOpType, IdentityBOp>;

    AOpType Aop(A);
    ArnoldiOpType arn(Aop, IdentityBOp());
    Arnoldi<Scalar, ArnoldiOpType> fac(arn, m);

    run_test(fac, A, m);
}

TEST_CASE("Lanczos factorization of Hermitian complex matrix", "[Lanczos]")
{
    std::srand(123);
    const int n = 10;
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(n, n);
    A = (A + A.adjoint()).eval();
    const int m = 6;

    using Scalar = std::complex<double>;
    using AOpType = DenseHermMatProd<Scalar>;
    using ArnoldiOpType = ArnoldiOp<Scalar, AOpType, IdentityBOp>;

    AOpType Aop(A);
    ArnoldiOpType arn(Aop, IdentityBOp());
    Lanczos<Scalar, ArnoldiOpType> fac(arn, m);

    run_test(fac, A, m);
}
