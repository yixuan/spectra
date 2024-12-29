// Test ../include/Spectra/LinAlg/Arnoldi.h, ../include/Spectra/LinAlg/Lanczos.h
#include <Eigen/Core>
#include <Spectra/LinAlg/Arnoldi.h>
#include <Spectra/LinAlg/Lanczos.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseGenMatProd.h>

using namespace Spectra;

#include "catch.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;

template <typename FacType>
void run_test(FacType &fac, const Matrix& A, int m)
{
    // Initialization
    const int n = A.rows();
    Vector v0 = Vector::Random(n);
    Eigen::Map<const Vector> v0map(v0.data(), n);
    Index op_counter = 0;
    fac.init(v0map, op_counter);

    // After initialization, the subspace dimension is one
    const int k1 = fac.subspace_dim();
    INFO("k1 = " << k1);
    REQUIRE(k1 == 1);

    // A*V = V*H + f*e
    constexpr double tol = 1e-12;
    Matrix V = fac.matrix_V().leftCols(k1);
    Matrix H = fac.matrix_H().topLeftCorner(k1, k1);
    Matrix resid = A * V - V * H;
    Vector f = fac.vector_f();
    INFO("AV - VH = \n" << resid);
    INFO("f = " << f.transpose());
    // Test residual
    REQUIRE((resid.rightCols<1>() - f).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

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
    REQUIRE((resid.rightCols<1>() - f).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Factorization to m
    fac.factorize_from(m / 2, m, op_counter);
    const int k3 = fac.subspace_dim();
    INFO("k3 = " << k3);
    REQUIRE(k3 == m);

    V = fac.matrix_V().leftCols(k3);
    H = fac.matrix_H().topLeftCorner(k3, k3);
    resid = A * V - V * H;
    f = fac.vector_f();
    Matrix VtV = V.transpose() * V;
    INFO("V = \n" << V);
    INFO("H = \n" << H);
    INFO("V'V = \n" << VtV);
    INFO("AV - VH = \n" << resid);
    INFO("f = " << f.transpose());
    // Test orthogonality of V
    Matrix iden = Matrix::Identity(m, m);
    REQUIRE((VtV - iden).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
    // Test residual
    REQUIRE(resid.leftCols(k3 - 1).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
    REQUIRE((resid.rightCols<1>() - f).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
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
