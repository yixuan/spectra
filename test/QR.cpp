// Test ../include/Spectra/LinAlg/UpperHessenbergQR.h and
//      ../include/Spectra/LinAlg/DoubleShiftQR.h
#include <complex>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Spectra/LinAlg/UpperHessenbergQR.h>
#include <Spectra/LinAlg/DoubleShiftQR.h>

#include "catch.hpp"

using namespace Spectra;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using MapMat = Eigen::Map<Matrix>;
using Complex = std::complex<double>;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexMapMat = Eigen::Map<ComplexMatrix>;

template <typename Solver, typename MatType>
void run_test(const MatType &H, typename MatType::Scalar shift)
{
    // In case MatType is a Map type, PlainMatrix is always the actual matrix type
    using PlainMatrix = typename MatType::PlainObject;
    using PlainVector = Eigen::Matrix<typename MatType::Scalar, Eigen::Dynamic, 1>;

    Solver decomp(H, shift);
    const int n = H.rows();
    constexpr double tol = 1e-12;
    // H - s * I
    PlainMatrix Hs = H - shift * PlainMatrix::Identity(n, n);

    // Obtain Q matrix
    PlainMatrix I = PlainMatrix::Identity(n, n);
    PlainMatrix Q = I;
    decomp.apply_QY(Q);

    // Test orthogonality
    PlainMatrix QtQ = Q.adjoint() * Q;
    INFO("||Q'Q - I||_inf = " << (QtQ - I).cwiseAbs().maxCoeff());
    REQUIRE((QtQ - I).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    PlainMatrix QQt = Q * Q.adjoint();
    INFO("||QQ' - I||_inf = " << (QQt - I).cwiseAbs().maxCoeff());
    REQUIRE((QQt - I).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Obtain R matrix and test whether it is upper triangular
    PlainMatrix R = decomp.matrix_R();
    PlainMatrix Rlower = R.template triangularView<Eigen::StrictlyLower>();
    INFO("Whether R is upper triangular, error = " << Rlower.cwiseAbs().maxCoeff());
    REQUIRE(Rlower.cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Compare Hs = H - s * I and QR
    INFO("||Hs - QR||_inf = " << (Hs - Q * R).cwiseAbs().maxCoeff());
    REQUIRE((Hs - Q * R).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Obtain Q'HQ
    PlainMatrix QtHQ_true = Q.adjoint() * H * Q;
    PlainMatrix QtHQ;
    decomp.matrix_QtHQ(QtHQ);
    INFO("max error of Q'HQ = " << (QtHQ - QtHQ_true).cwiseAbs().maxCoeff());
    REQUIRE((QtHQ - QtHQ_true).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Test "apply" functions
    PlainMatrix Y = PlainMatrix::Random(n, n);

    PlainMatrix QY = Y;
    decomp.apply_QY(QY);
    INFO("max error of QY = " << (QY - Q * Y).cwiseAbs().maxCoeff());
    REQUIRE((QY - Q * Y).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    PlainMatrix YQ = Y;
    decomp.apply_YQ(YQ);
    INFO("max error of YQ = " << (YQ - Y * Q).cwiseAbs().maxCoeff());
    REQUIRE((YQ - Y * Q).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    PlainMatrix QtY = Y;
    decomp.apply_QtY(QtY);
    INFO("max error of Q'Y = " << (QtY - Q.adjoint() * Y).cwiseAbs().maxCoeff());
    REQUIRE((QtY - Q.adjoint() * Y).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    PlainMatrix YQt = Y;
    decomp.apply_YQt(YQt);
    INFO("max error of YQ' = " << (YQt - Y * Q.adjoint()).cwiseAbs().maxCoeff());
    REQUIRE((YQt - Y * Q.adjoint()).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Test "apply" functions for vectors
    PlainVector y = PlainVector::Random(n);

    PlainVector Qy = y;
    decomp.apply_QY(Qy);
    INFO("max error of Qy = " << (Qy - Q * y).cwiseAbs().maxCoeff());
    REQUIRE((Qy - Q * y).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    PlainVector Qty = y;
    decomp.apply_QtY(Qty);
    INFO("max error of Q'y = " << (Qty - Q.adjoint() * y).cwiseAbs().maxCoeff());
    REQUIRE((Qty - Q.adjoint() * y).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
}

TEST_CASE("QR of real upper Hessenberg matrix", "[QR]")
{
    std::srand(123);
    const int n = 100;
    Matrix M = Matrix::Random(n, n);
    Matrix H = M.triangularView<Eigen::Upper>();
    H.diagonal(-1) = M.diagonal(-1);

    run_test<UpperHessenbergQR<double>>(H, 1.2345);

    MapMat Hmap(H.data(), H.rows(), H.cols());
    run_test<UpperHessenbergQR<double>>(Hmap, 0.6789);
}

TEST_CASE("QR of real tridiagonal matrix", "[QR]")
{
    std::srand(123);
    const int n = 100;
    Matrix M = Matrix::Random(n, n);
    Matrix H = Matrix::Zero(n, n);
    H.diagonal() = M.diagonal();
    H.diagonal(-1) = M.diagonal(-1);
    H.diagonal(1) = M.diagonal(-1);

    run_test<TridiagQR<double>>(H, 1.2345);

    MapMat Hmap(H.data(), H.rows(), H.cols());
    run_test<TridiagQR<double>>(Hmap, 0.6789);
}

TEST_CASE("QR decomposition with double shifts", "[QR]")
{
    std::srand(123);
    const int n = 100;
    constexpr double tol = 1e-12;

    Matrix M = Matrix::Random(n, n);
    Matrix H = M.triangularView<Eigen::Upper>();
    H.diagonal(-1) = M.diagonal(-1);
    H(1, 0) = 0;  // Test for the case when sub-diagonal element is zero

    const double s = 2, t = 3;

    Matrix Hst = H * H - s * H + t * Matrix::Identity(n, n);
    Eigen::HouseholderQR<Matrix> qr(Hst);
    Matrix Q0 = qr.householderQ();

    DoubleShiftQR<double> decomp(H, s, t);
    Matrix Q = Matrix::Identity(n, n);
    decomp.apply_YQ(Q);

    // Equal up to signs
    INFO("max error of Q = " << (Q.cwiseAbs() - Q0.cwiseAbs()).cwiseAbs().maxCoeff());
    REQUIRE((Q.cwiseAbs() - Q0.cwiseAbs()).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Test Q'HQ
    Matrix QtHQ;
    decomp.matrix_QtHQ(QtHQ);
    INFO("max error of Q'HQ = " << (QtHQ - Q.transpose() * H * Q).cwiseAbs().maxCoeff());
    REQUIRE((QtHQ - Q.transpose() * H * Q).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // Test apply functions
    Vector y = Vector::Random(n);
    Matrix Y = Matrix::Random(n / 2, n);

    Vector Qty = y;
    decomp.apply_QtY(Qty);
    INFO("max error of Q'y = " << (Qty - Q.transpose() * y).cwiseAbs().maxCoeff());
    REQUIRE((Qty - Q.transpose() * y).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    Matrix YQ = Y;
    decomp.apply_YQ(YQ);
    INFO("max error of YQ = " << (YQ - Y * Q).cwiseAbs().maxCoeff());
    REQUIRE((YQ - Y * Q).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));
}

TEST_CASE("QR of complex upper Hessenberg matrix", "[QR]")
{
    std::srand(123);
    const int n = 100;
    ComplexMatrix M = ComplexMatrix::Random(n, n);
    ComplexMatrix H = M.triangularView<Eigen::Upper>();
    H.diagonal(-1) = M.diagonal(-1);

    run_test<UpperHessenbergQR<Complex>>(H, Complex(1.2345, -5.4321));

    ComplexMapMat Hmap(H.data(), H.rows(), H.cols());
    run_test<UpperHessenbergQR<Complex>>(Hmap, Complex(0.6789, -0.9876));
}
