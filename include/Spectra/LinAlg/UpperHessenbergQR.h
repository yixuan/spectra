// Copyright (C) 2016-2025 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_UPPER_HESSENBERG_QR_H
#define SPECTRA_UPPER_HESSENBERG_QR_H

#include <Eigen/Core>
#include <cmath>      // std::abs
#include <complex>    // std::complex
#include <algorithm>  // std::fill
#include <stdexcept>  // std::invalid_argument, std::logic_error

#include "../Util/TypeTraits.h"
#include "Givens.h"

namespace Spectra {

///
/// \defgroup Internals Internal Classes
///
/// Classes for internal use. May be useful to developers.
///

///
/// \ingroup Internals
/// @{
///

///
/// \defgroup LinearAlgebra Linear Algebra
///
/// A number of classes for linear algebra operations.
///

///
/// \ingroup LinearAlgebra
///
/// Perform the QR decomposition of an upper Hessenberg matrix.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float`, `double` and `long double`.
///
template <typename Scalar = double>
class UpperHessenbergQR
{
private:
    // The type of the real part, e.g.,
    //     Scalar = double               => RealScalar = double
    //     Scalar = std::complex<double> => RealScalar = double
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using RealArray = Eigen::Array<RealScalar, Eigen::Dynamic, 1>;

    using GenericMatrix = Eigen::Ref<Matrix>;
    using ConstGenericMatrix = const Eigen::Ref<const Matrix>;

    Matrix m_mat_R;

protected:
    Index m_n;
    Scalar m_shift;
    // Given an upper Hessenberg matrix H, find an orthogonal matrix Q
    // and upper triangular matrix R such that H - s*I = QR
    // In other words, we want to transform H - s*I to R by
    // orthogonal transformation Q^H (H - s*I) = R
    // Q can be represented by a sequence of Givens rotations,
    // Q = Q1 * Q2 * ... * Q_{n-1}, and each Qi is determined by the 2x2 matrix
    //     Gi = [  c[i]   s[i]]
    //          [-(s[i])  c[i]]
    // where c is real, and (s) is the conjugate of s
    //
    // Each Gi is responsible for transforming a 2x1 vector u = [x] into v = [r]
    //                                                          [y]          [0]
    // See Givens.h for details
    RealArray m_rot_cos;
    Array m_rot_sin;
    bool m_computed;

public:
    ///
    /// Constructor to preallocate memory. Computation can
    /// be performed later by calling the compute() method.
    ///
    UpperHessenbergQR(Index size) :
        m_n(size),
        m_shift(0),
        m_rot_cos(m_n - 1),
        m_rot_sin(m_n - 1),
        m_computed(false)
    {}

    ///
    /// Constructor to create an object that performs and stores the
    /// QR decomposition of an upper Hessenberg matrix `mat`, with an
    /// optional shift: \f$H-sI=QR\f$. Here \f$H\f$ stands for the matrix
    /// `mat`, and \f$s\f$ is the shift.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the upper triangular and the subdiagonal elements of
    /// the matrix are used.
    ///
    UpperHessenbergQR(ConstGenericMatrix& mat, const Scalar& shift = Scalar(0)) :
        m_n(mat.rows()),
        m_shift(shift),
        m_rot_cos(m_n - 1),
        m_rot_sin(m_n - 1),
        m_computed(false)
    {
        compute(mat, shift);
    }

    ///
    /// Virtual destructor.
    ///
    virtual ~UpperHessenbergQR() {}

    ///
    /// Compute the QR decomposition of an upper Hessenberg matrix with
    /// an optional shift.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the upper triangular and the subdiagonal elements of
    /// the matrix are used.
    ///
    virtual void compute(ConstGenericMatrix& mat, const Scalar& shift = Scalar(0))
    {
        m_n = mat.rows();
        if (m_n != mat.cols())
            throw std::invalid_argument("UpperHessenbergQR: matrix must be square");

        m_shift = shift;
        m_mat_R.resize(m_n, m_n);
        m_rot_cos.resize(m_n - 1);
        m_rot_sin.resize(m_n - 1);

        // Make a copy of mat - s * I
        m_mat_R.noalias() = mat;
        m_mat_R.diagonal().array() -= m_shift;

        Scalar xi, xj, r, s;
        RealScalar c;
        Scalar *Rii, *ptr;
        const Index n1 = m_n - 1;
        for (Index i = 0; i < n1; i++)
        {
            Rii = &m_mat_R.coeffRef(i, i);

            // Make sure R is upper Hessenberg
            // Zero the elements below R[i + 1, i]
            std::fill(Rii + 2, Rii + m_n - i, Scalar(0));

            xi = Rii[0];  // R[i, i]
            xj = Rii[1];  // R[i + 1, i]
            Givens<Scalar>::compute_rotation(xi, xj, r, c, s);
            m_rot_cos.coeffRef(i) = c;
            m_rot_sin.coeffRef(i) = s;

            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [  c   s]
            //     [-(s)  c]
            // and then do R[i:(i + 1), i:(n - 1)] = G^H * R[i:(i + 1), i:(n - 1)]

            // GH << c, -s, Eigen::numext::conj(s), c;
            // m_mat_R.block(i, i, 2, m_n - i) = GH * m_mat_R.block(i, i, 2, m_n - i);
            Rii[0] = r;       // R[i, i]     => r
            Rii[1] = 0;       // R[i + 1, i] => 0
            ptr = Rii + m_n;  // R[i, k], k = i+1, i+2, ..., n-1
            for (Index j = i + 1; j < m_n; j++, ptr += m_n)
            {
                const Scalar tmp = ptr[0];
                ptr[0] = c * tmp - s * ptr[1];
                ptr[1] = Eigen::numext::conj(s) * tmp + c * ptr[1];
            }

            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required
            // In such case we only update R[i + 1, (i + 1):(n - 1)]
            // m_mat_R.block(i + 1, i + 1, 1, m_n - i - 1) *= c;
            // m_mat_R.block(i + 1, i + 1, 1, m_n - i - 1) += conj(s) * m_mat_R.block(i, i + 1, 1, m_n - i - 1);
        }

        m_computed = true;
    }

    ///
    /// Return the \f$R\f$ matrix in the QR decomposition, which is an
    /// upper triangular matrix.
    ///
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    virtual Matrix matrix_R() const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        return m_mat_R;
    }

    ///
    /// Overwrite `dest` with \f$Q^H H Q = RQ + sI\f$, where \f$H\f$ is the input matrix `mat`,
    /// and \f$s\f$ is the shift. The result is an upper Hessenberg matrix.
    ///
    /// \param mat The matrix to be overwritten, whose type should be `Eigen::Matrix<Scalar, ...>`,
    /// depending on the template parameter `Scalar` defined.
    ///
    virtual void matrix_QtHQ(Matrix& dest) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        // Make a copy of the R matrix
        dest.resize(m_n, m_n);
        dest.noalias() = m_mat_R;

        // Compute the RQ matrix
        const Index n1 = m_n - 1;
        for (Index i = 0; i < n1; i++)
        {
            const RealScalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // RQ[, i:(i + 1)] = RQ[, i:(i + 1)] * Gi
            // Gi = [  c[i]   s[i]]
            //      [-(s[i])  c[i]]
            Scalar *Yi, *Yi1;
            Yi = &dest.coeffRef(0, i);
            Yi1 = Yi + m_n;  // RQ[0, i + 1]
            const Index i2 = i + 2;
            for (Index j = 0; j < i2; j++)
            {
                const Scalar tmp = Yi[j];
                Yi[j] = c * tmp - Eigen::numext::conj(s) * Yi1[j];
                Yi1[j] = s * tmp + c * Yi1[j];
            }

            /* Vector dest = RQ.block(0, i, i + 2, 1);
            dest.block(0, i, i + 2, 1)     = c * Yi - conj(s) * dest.block(0, i + 1, i + 2, 1);
            dest.block(0, i + 1, i + 2, 1) = s * Yi + c * dest.block(0, i + 1, i + 2, 1); */
        }

        // Add the shift to the diagonal
        dest.diagonal().array() += m_shift;
    }

    ///
    /// Apply the \f$Q\f$ matrix to a vector \f$y\f$.
    ///
    /// \param Y A vector that will be overwritten by the matrix product \f$Qy\f$.
    ///
    /// Vector type can be `Eigen::Vector<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    // Y -> QY = G1 * G2 * ... * Y
    void apply_QY(Vector& Y) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        for (Index i = m_n - 2; i >= 0; i--)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1)] = Gi * Y[i:(i + 1)]
            // Gi = [  c[i]   s[i]]
            //      [-(s[i])  c[i]]
            const Scalar tmp = Y[i];
            Y[i] = c * tmp + s * Y[i + 1];
            Y[i + 1] = -Eigen::numext::conj(s) * tmp + c * Y[i + 1];
        }
    }

    ///
    /// Apply the \f$Q\f$ matrix to a vector \f$y\f$.
    ///
    /// \param Y A vector that will be overwritten by the matrix product \f$Q'y\f$.
    ///
    /// Vector type can be `Eigen::Vector<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    // Y -> Q^H Y = G_{n-1}^H * ... * G2^H * G1^H * Y
    void apply_QtY(Vector& Y) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        const Index n1 = m_n - 1;
        for (Index i = 0; i < n1; i++)
        {
            const RealScalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1)] = Gi^H * Y[i:(i + 1)]
            // Gi = [  c[i]   s[i]], Gi^H = [ c[i]  -s[i]]
            //      [-(s[i])  c[i]]         [(s[i])  c[i]]
            const Scalar tmp = Y[i];
            Y[i] = c * tmp - s * Y[i + 1];
            Y[i + 1] = Eigen::numext::conj(s) * tmp + c * Y[i + 1];
        }
    }

    ///
    /// Apply the \f$Q\f$ matrix to another matrix \f$Y\f$.
    ///
    /// \param Y A matrix that will be overwritten by the matrix product \f$QY\f$.
    ///
    /// Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    ///
    // Y -> QY = G1 * G2 * ... * Y
    void apply_QY(GenericMatrix Y) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        RowVector Yi(Y.cols()), Yi1(Y.cols());
        for (Index i = m_n - 2; i >= 0; i--)
        {
            const RealScalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1), ] = Gi * Y[i:(i + 1), ]
            // Gi = [  c[i]   s[i]]
            //      [-(s[i])  c[i]]
            Yi.noalias() = Y.row(i);
            Yi1.noalias() = Y.row(i + 1);
            Y.row(i) = c * Yi + s * Yi1;
            Y.row(i + 1) = -Eigen::numext::conj(s) * Yi + c * Yi1;
        }
    }

    ///
    /// Apply the \f$Q\f$ matrix to another matrix \f$Y\f$.
    ///
    /// \param Y A matrix that will be overwritten by the matrix product \f$Q'Y\f$.
    ///
    /// Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    ///
    // Y -> Q^H Y = G_{n-1}^H * ... * G2^H * G1^H * Y
    void apply_QtY(GenericMatrix Y) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        RowVector Yi(Y.cols()), Yi1(Y.cols());
        const Index n1 = m_n - 1;
        for (Index i = 0; i < n1; i++)
        {
            const RealScalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1), ] = Gi^H * Y[i:(i + 1), ]
            // Gi = [  c[i]   s[i]], Gi^H = [ c[i]  -s[i]]
            //      [-(s[i])  c[i]]         [(s[i])  c[i]]
            Yi.noalias() = Y.row(i);
            Yi1.noalias() = Y.row(i + 1);
            Y.row(i) = c * Yi - s * Yi1;
            Y.row(i + 1) = Eigen::numext::conj(s) * Yi + c * Yi1;
        }
    }

    ///
    /// Apply the \f$Q\f$ matrix to another matrix \f$Y\f$.
    ///
    /// \param Y A matrix that will be overwritten by the matrix product \f$YQ\f$.
    ///
    /// Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    ///
    // Y -> YQ = Y * G1 * G2 * ...
    void apply_YQ(GenericMatrix Y) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        /*Vector Yi(Y.rows());
        for(Index i = 0; i < m_n - 1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi
            // Gi = [  c[i]   s[i]]
            //      [-(s[i])  c[i]]
            Yi.noalias() = Y.col(i);
            Y.col(i)     = c * Yi - conj(s) * Y.col(i + 1);
            Y.col(i + 1) = s * Yi + c * Y.col(i + 1);
        }*/
        Scalar *Y_col_i, *Y_col_i1;
        const Index n1 = m_n - 1;
        const Index nrow = Y.rows();
        for (Index i = 0; i < n1; i++)
        {
            const RealScalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);

            Y_col_i = &Y.coeffRef(0, i);
            Y_col_i1 = &Y.coeffRef(0, i + 1);
            for (Index j = 0; j < nrow; j++)
            {
                Scalar tmp = Y_col_i[j];
                Y_col_i[j] = c * tmp - Eigen::numext::conj(s) * Y_col_i1[j];
                Y_col_i1[j] = s * tmp + c * Y_col_i1[j];
            }
        }
    }

    ///
    /// Apply the \f$Q\f$ matrix to another matrix \f$Y\f$.
    ///
    /// \param Y A matrix that will be overwritten by the matrix product \f$YQ'\f$.
    ///
    /// Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    ///
    // Y -> YQ^H = Y * G_{n-1}^H * ... * G2^H * G1^H
    void apply_YQt(GenericMatrix Y) const
    {
        if (!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        Vector Yi(Y.rows());
        for (Index i = m_n - 2; i >= 0; i--)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi^H
            // Gi = [  c[i]   s[i]], Gi^H = [ c[i]  -s[i]]
            //      [-(s[i])  c[i]]         [(s[i])  c[i]]
            Yi.noalias() = Y.col(i);
            Y.col(i) = c * Yi + Eigen::numext::conj(s) * Y.col(i + 1);
            Y.col(i + 1) = -s * Yi + c * Y.col(i + 1);
        }
    }
};

///
/// \ingroup LinearAlgebra
///
/// Perform the QR decomposition of a tridiagonal matrix, a special
/// case of upper Hessenberg matrices.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float`, `double` and `long double`.
///
template <typename Scalar = double>
class TridiagQR : public UpperHessenbergQR<Scalar>
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstGenericMatrix = const Eigen::Ref<const Matrix>;
    using ComplexMatrix = Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>;

    using UpperHessenbergQR<Scalar>::m_n;
    using UpperHessenbergQR<Scalar>::m_shift;
    using UpperHessenbergQR<Scalar>::m_rot_cos;
    using UpperHessenbergQR<Scalar>::m_rot_sin;
    using UpperHessenbergQR<Scalar>::m_computed;

    Vector m_T_diag;   // diagonal elements of T
    Vector m_T_subd;   // 1st subdiagonal of T
    Vector m_R_diag;   // diagonal elements of R, where T = QR
    Vector m_R_supd;   // 1st superdiagonal of R
    Vector m_R_supd2;  // 2nd superdiagonal of R

public:
    ///
    /// Constructor to preallocate memory. Computation can
    /// be performed later by calling the compute() method.
    ///
    TridiagQR(Index size) :
        UpperHessenbergQR<Scalar>(size)
    {}

    ///
    /// Constructor to create an object that performs and stores the
    /// QR decomposition of a tridiagonal matrix `mat`, with an
    /// optional shift: \f$T-sI=QR\f$. Here \f$T\f$ stands for the matrix
    /// `mat`, and \f$s\f$ is the shift.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the diagonal and subdiagonal elements of the matrix are used.
    ///
    TridiagQR(ConstGenericMatrix& mat, const Scalar& shift = Scalar(0)) :
        UpperHessenbergQR<Scalar>(mat.rows())
    {
        this->compute(mat, shift);
    }

    ///
    /// Compute the QR decomposition of a tridiagonal matrix with an
    /// optional shift.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the diagonal and subdiagonal elements of the matrix are used.
    ///
    void compute(ConstGenericMatrix& mat, const Scalar& shift = Scalar(0)) override
    {
        using std::abs;

        m_n = mat.rows();
        if (m_n != mat.cols())
            throw std::invalid_argument("TridiagQR: matrix must be square");

        m_shift = shift;
        m_rot_cos.resize(m_n - 1);
        m_rot_sin.resize(m_n - 1);

        // Save the diagonal and subdiagonal elements of T
        m_T_diag.resize(m_n);
        m_T_subd.resize(m_n - 1);
        m_T_diag.noalias() = mat.diagonal();
        m_T_subd.noalias() = mat.diagonal(-1);

        // Deflation of small sub-diagonal elements
        const Scalar eps = TypeTraits<Scalar>::epsilon();
        for (Index i = 0; i < m_n - 1; i++)
        {
            if (abs(m_T_subd[i]) <= eps * (abs(m_T_diag[i]) + abs(m_T_diag[i + 1])))
                m_T_subd[i] = Scalar(0);
        }

        // Apply shift and copy T to R
        m_R_diag.resize(m_n);
        m_R_supd.resize(m_n - 1);
        m_R_supd2.resize(m_n - 2);
        m_R_diag.array() = m_T_diag.array() - m_shift;
        m_R_supd.noalias() = m_T_subd;

        // A number of pointers to avoid repeated address calculation
        Scalar *c = m_rot_cos.data(),  // pointer to the cosine vector
            *s = m_rot_sin.data(),     // pointer to the sine vector
            r;
        const Index n1 = m_n - 1, n2 = m_n - 2;
        for (Index i = 0; i < n1; i++)
        {
            // Rdiag[i] == R[i, i]
            // Tsubd[i] == R[i + 1, i]
            // r = sqrt(R[i, i]^2 + R[i + 1, i]^2)
            // c = R[i, i] / r, s = -R[i + 1, i] / r
            Givens<Scalar>::compute_rotation(m_R_diag.coeff(i), m_T_subd.coeff(i), r, *c, *s);

            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do R[i:(i + 1), i:(i + 2)] = G' * R[i:(i + 1), i:(i + 2)]

            // Update R[i, i] and R[i + 1, i]
            // The updated value of R[i, i] is known to be r
            // The updated value of R[i + 1, i] is known to be 0
            m_R_diag.coeffRef(i) = r;
            // Update R[i, i + 1] and R[i + 1, i + 1]
            // Rsupd[i] == R[i, i + 1]
            // Rdiag[i + 1] == R[i + 1, i + 1]
            const Scalar Tii1 = m_R_supd.coeff(i);
            const Scalar Ti1i1 = m_R_diag.coeff(i + 1);
            m_R_supd.coeffRef(i) = (*c) * Tii1 - (*s) * Ti1i1;
            m_R_diag.coeffRef(i + 1) = (*s) * Tii1 + (*c) * Ti1i1;
            // Update R[i, i + 2] and R[i + 1, i + 2]
            // Rsupd2[i] == R[i, i + 2]
            // Rsupd[i + 1] == R[i + 1, i + 2]
            if (i < n2)
            {
                m_R_supd2.coeffRef(i) = -(*s) * m_R_supd.coeff(i + 1);
                m_R_supd.coeffRef(i + 1) *= (*c);
            }

            c++;
            s++;

            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update R[i + 1, (i + 1):(i + 2)]
            // R[i + 1, i + 1] = c * R[i + 1, i + 1] + s * R[i, i + 1];
            // R[i + 1, i + 2] *= c;
        }

        m_computed = true;
    }

    ///
    /// Return the \f$R\f$ matrix in the QR decomposition, which is an
    /// upper triangular matrix.
    ///
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Matrix matrix_R() const override
    {
        if (!m_computed)
            throw std::logic_error("TridiagQR: need to call compute() first");

        Matrix R = Matrix::Zero(m_n, m_n);
        R.diagonal().noalias() = m_R_diag;
        R.diagonal(1).noalias() = m_R_supd;
        R.diagonal(2).noalias() = m_R_supd2;

        return R;
    }

    ///
    /// Overwrite `dest` with \f$Q'TQ = RQ + sI\f$, where \f$T\f$ is the input matrix `mat`,
    /// and \f$s\f$ is the shift. The result is a tridiagonal matrix.
    ///
    /// \param mat The matrix to be overwritten, whose type should be `Eigen::Matrix<Scalar, ...>`,
    /// depending on the template parameter `Scalar` defined.
    ///
    void matrix_QtHQ(Matrix& dest) const override
    {
        using std::abs;

        if (!m_computed)
            throw std::logic_error("TridiagQR: need to call compute() first");

        // In exact arithmetics, Q'TQ = RQ + sI, so we can just apply Q to R and add the shift.
        // However, some numerical examples show that this algorithm decreases the precision,
        // so we directly apply Q' and Q to T.

        // Copy the saved diagonal and subdiagonal elements of T to `dest`
        dest.resize(m_n, m_n);
        dest.setZero();
        dest.diagonal().noalias() = m_T_diag;
        dest.diagonal(-1).noalias() = m_T_subd;

        // Ti = [x  y  0],  Gi = [ cos[i]  sin[i]  0],  Gi' * Ti * Gi = [x'  y'  o']
        //      [y  z  w]        [-sin[i]  cos[i]  0]                   [y'  z'  w']
        //      [0  w  u]        [      0       0  1]                   [o'  w'  u']
        //
        // x' = c2*x - 2*c*s*y + s2*z
        // y' = c*s*(x-z) + (c2-s2)*y
        // z' = s2*x + 2*c*s*y + c2*z
        // o' = -s*w, w' = c*w, u' = u
        //
        // In iteration (i+1), (y', o') will be further updated to (y'', o''),
        // where o'' = 0, y'' = cos[i+1]*y' - sin[i+1]*o'
        const Index n1 = m_n - 1, n2 = m_n - 2;
        for (Index i = 0; i < n1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            const Scalar cs = c * s, c2 = c * c, s2 = s * s;
            const Scalar x = dest.coeff(i, i),
                         y = dest.coeff(i + 1, i),
                         z = dest.coeff(i + 1, i + 1);
            const Scalar c2x = c2 * x, s2x = s2 * x, c2z = c2 * z, s2z = s2 * z;
            const Scalar csy2 = Scalar(2) * c * s * y;

            // Update the diagonal and the lower subdiagonal of dest
            dest.coeffRef(i, i) = c2x - csy2 + s2z;                  // x'
            dest.coeffRef(i + 1, i) = cs * (x - z) + (c2 - s2) * y;  // y'
            dest.coeffRef(i + 1, i + 1) = s2x + csy2 + c2z;          // z'

            if (i < n2)
            {
                const Scalar ci1 = m_rot_cos.coeff(i + 1);
                const Scalar si1 = m_rot_sin.coeff(i + 1);
                const Scalar o = -s * m_T_subd.coeff(i + 1);                     // o'
                dest.coeffRef(i + 2, i + 1) *= c;                                // w'
                dest.coeffRef(i + 1, i) = ci1 * dest.coeff(i + 1, i) - si1 * o;  // y''
            }
        }

        // Deflation of small sub-diagonal elements
        const Scalar eps = TypeTraits<Scalar>::epsilon();
        for (Index i = 0; i < n1; i++)
        {
            const Scalar diag = abs(dest.coeff(i, i)) + abs(dest.coeff(i + 1, i + 1));
            if (abs(dest.coeff(i + 1, i)) <= eps * diag)
                dest.coeffRef(i + 1, i) = Scalar(0);
        }

        // Copy the lower subdiagonal to upper subdiagonal
        dest.diagonal(1).noalias() = dest.diagonal(-1);
    }

    ///
    /// The version of matrix_QtHQ() when `dest` has a complex value type.
    ///
    /// This is used in Hermitian eigen solvers where the result is stored
    /// as a complex matrix.
    ///
    void matrix_QtHQ(ComplexMatrix& dest) const
    {
        // Simply compute the real-typed result and copy to the complex one
        Matrix dest_real;
        this->matrix_QtHQ(dest_real);
        dest.resize(m_n, m_n);
        dest.noalias() = dest_real.template cast<std::complex<Scalar>>();
    }
};

///
/// @}
///

}  // namespace Spectra

#endif  // SPECTRA_UPPER_HESSENBERG_QR_H
