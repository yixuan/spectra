// Copyright (C) 2016-2018 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef UPPER_HESSENBERG_QR_H
#define UPPER_HESSENBERG_QR_H

#include <Eigen/Core>
#include <cmath>      // std::sqrt
#include <algorithm>  // std::fill, std::copy
#include <stdexcept>  // std::logic_error

namespace Spectra {


///
/// \defgroup LinearAlgebra Linear Algebra
///
/// A number of classes for linear algebra operations.

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
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, 2, 2> Matrix22;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

    typedef typename Matrix::Index Index;

    typedef Eigen::Ref<Matrix> GenericMatrix;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

protected:
    Index m_n;
    Matrix m_mat_T;
    // Gi = [ cos[i]  sin[i]]
    //      [-sin[i]  cos[i]]
    // Q = G1 * G2 * ... * G_{n-1}
    Array m_rot_cos;
    Array m_rot_sin;
    bool m_computed;

    // Given x and y, compute 1) r = sqrt(x^2 + y^2), 2) c = x / r, 3) s = -y / r
    // If both x and y are zero, set c = 1 and s = 0
    // We must implement it in a numerically stable way
    static void compute_rotation(const Scalar& x, const Scalar& y, Scalar& r, Scalar& c, Scalar& s)
    {
        using std::sqrt;

        const Scalar xsign = (x > Scalar(0)) - (x < Scalar(0));
        const Scalar ysign = (y > Scalar(0)) - (y < Scalar(0));
        const Scalar xabs = x * xsign;
        const Scalar yabs = y * ysign;
        if(xabs > yabs)
        {
            // In this case xabs != 0
            const Scalar ratio = yabs / xabs;  // so that 0 <= ratio < 1
            const Scalar common = sqrt(Scalar(1) + ratio * ratio);
            c = xsign / common;
            r = xabs * common;
            s = -y / r;
        } else {
            if(yabs == Scalar(0))
            {
                r = Scalar(0); c = Scalar(1); s = Scalar(0);
                return;
            }
            const Scalar ratio = xabs / yabs;  // so that 0 <= ratio <= 1
            const Scalar common = sqrt(Scalar(1) + ratio * ratio);
            s = -ysign / common;
            r = yabs * common;
            c = x / r;
        }
    }

public:
    ///
    /// Default constructor. Computation can
    /// be performed later by calling the compute() method.
    ///
    UpperHessenbergQR() :
        m_n(0), m_computed(false)
    {}

    ///
    /// Constructor to create an object that performs and stores the
    /// QR decomposition of an upper Hessenberg matrix `mat`.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the upper triangular and the lower subdiagonal parts of
    /// the matrix are used.
    ///
    UpperHessenbergQR(ConstGenericMatrix& mat) :
        m_n(mat.rows()),
        m_mat_T(m_n, m_n),
        m_rot_cos(m_n - 1),
        m_rot_sin(m_n - 1),
        m_computed(false)
    {
        compute(mat);
    }

    ///
    /// We have virtual functions, so need a virtual destructor
    ///
    virtual ~UpperHessenbergQR(){};

    ///
    /// Conduct the QR factorization of an upper Hessenberg matrix.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the upper triangular and the lower subdiagonal parts of
    /// the matrix are used.
    ///
    virtual void compute(ConstGenericMatrix& mat)
    {
        m_n = mat.rows();
        if(m_n != mat.cols())
            throw std::invalid_argument("UpperHessenbergQR: matrix must be square");

        m_mat_T.resize(m_n, m_n);
        m_rot_cos.resize(m_n - 1);
        m_rot_sin.resize(m_n - 1);

        // Make a copy of mat
        std::copy(mat.data(), mat.data() + mat.size(), m_mat_T.data());

        Scalar xi, xj, r, c, s;
        Scalar *Tii, *ptr;
        const Index n1 = m_n - 1;
        for(Index i = 0; i < n1; i++)
        {
            Tii = &m_mat_T.coeffRef(i, i);

            // Make sure mat_T is upper Hessenberg
            // Zero the elements below mat_T(i + 1, i)
            std::fill(Tii + 2, Tii + m_n - i, Scalar(0));

            xi = Tii[0];  // mat_T(i, i)
            xj = Tii[1];  // mat_T(i + 1, i)
            compute_rotation(xi, xj, r, c, s);
            m_rot_cos[i] = c;
            m_rot_sin[i] = s;

            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(n - 1)] = G' * T[i:(i + 1), i:(n - 1)]

            // Gt << c, -s, s, c;
            // m_mat_T.block(i, i, 2, m_n - i) = Gt * m_mat_T.block(i, i, 2, m_n - i);
            Tii[0] = r;  // m_mat_T(i, i)     => r
            Tii[1] = 0;  // m_mat_T(i + 1, i) => 0
            ptr = Tii + m_n; // m_mat_T(i, k), k = i+1, i+2, ..., n-1
            for(Index j = i + 1; j < m_n; j++, ptr += m_n)
            {
                Scalar tmp = ptr[0];
                ptr[0] = c * tmp - s * ptr[1];
                ptr[1] = s * tmp + c * ptr[1];
            }

            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update T[i + 1, (i + 1):(n - 1)]
            // m_mat_T.block(i + 1, i + 1, 1, m_n - i - 1) *= c;
            // m_mat_T.block(i + 1, i + 1, 1, m_n - i - 1) += s * mat_T.block(i, i + 1, 1, m_n - i - 1);
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
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        return m_mat_T;
    }

    ///
    /// Return the \f$RQ\f$ matrix, the multiplication of \f$R\f$ and \f$Q\f$,
    /// which is an upper Hessenberg matrix.
    ///
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    virtual Matrix matrix_RQ() const
    {
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        // Make a copy of the R matrix
        Matrix RQ = m_mat_T.template triangularView<Eigen::Upper>();

        const Index n1 = m_n - 1;
        for(Index i = 0; i < n1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // RQ[, i:(i + 1)] = RQ[, i:(i + 1)] * Gi
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Scalar *Yi, *Yi1;
            Yi = &RQ.coeffRef(0, i);
            Yi1 = Yi + m_n;  // RQ(0, i + 1)
            const Index i2 = i + 2;
            for(Index j = 0; j < i2; j++)
            {
                const Scalar tmp = Yi[j];
                Yi[j]  = c * tmp - s * Yi1[j];
                Yi1[j] = s * tmp + c * Yi1[j];
            }

            /* Vector Yi = RQ.block(0, i, i + 2, 1);
            RQ.block(0, i, i + 2, 1)     = c * Yi - s * RQ.block(0, i + 1, i + 2, 1);
            RQ.block(0, i + 1, i + 2, 1) = s * Yi + c * RQ.block(0, i + 1, i + 2, 1); */
        }

        return RQ;
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
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        for(Index i = m_n - 2; i >= 0; i--)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1)] = Gi * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            const Scalar tmp = Y[i];
            Y[i]     =  c * tmp + s * Y[i + 1];
            Y[i + 1] = -s * tmp + c * Y[i + 1];
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
    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    void apply_QtY(Vector& Y) const
    {
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        const Index n1 = m_n - 1;
        for(Index i = 0; i < n1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1)] = Gi' * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            const Scalar tmp = Y[i];
            Y[i]     = c * tmp - s * Y[i + 1];
            Y[i + 1] = s * tmp + c * Y[i + 1];
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
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        RowVector Yi(Y.cols()), Yi1(Y.cols());
        for(Index i = m_n - 2; i >= 0; i--)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1), ] = Gi * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi.noalias()  = Y.row(i);
            Yi1.noalias() = Y.row(i + 1);
            Y.row(i)      =  c * Yi + s * Yi1;
            Y.row(i + 1)  = -s * Yi + c * Yi1;
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
    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    void apply_QtY(GenericMatrix Y) const
    {
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        RowVector Yi(Y.cols()), Yi1(Y.cols());
        const Index n1 = m_n - 1;
        for(Index i = 0; i < n1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[i:(i + 1), ] = Gi' * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi.noalias()  = Y.row(i);
            Yi1.noalias() = Y.row(i + 1);
            Y.row(i)      = c * Yi - s * Yi1;
            Y.row(i + 1)  = s * Yi + c * Yi1;
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
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        /*Vector Yi(Y.rows());
        for(Index i = 0; i < m_n - 1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi.noalias() = Y.col(i);
            Y.col(i)     = c * Yi - s * Y.col(i + 1);
            Y.col(i + 1) = s * Yi + c * Y.col(i + 1);
        }*/
        Scalar *Y_col_i, *Y_col_i1;
        const Index n1 = m_n - 1;
        const Index nrow = Y.rows();
        for(Index i = 0; i < n1; i++)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);

            Y_col_i  = &Y.coeffRef(0, i);
            Y_col_i1 = &Y.coeffRef(0, i + 1);
            for(Index j = 0; j < nrow; j++)
            {
                Scalar tmp = Y_col_i[j];
                Y_col_i[j]  = c * tmp - s * Y_col_i1[j];
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
    // Y -> YQ' = Y * G_{n-1}' * ... * G2' * G1'
    void apply_YQt(GenericMatrix Y) const
    {
        if(!m_computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");

        Vector Yi(Y.rows());
        for(Index i = m_n - 2; i >= 0; i--)
        {
            const Scalar c = m_rot_cos.coeff(i);
            const Scalar s = m_rot_sin.coeff(i);
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi'
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi.noalias() = Y.col(i);
            Y.col(i)     =  c * Yi + s * Y.col(i + 1);
            Y.col(i + 1) = -s * Yi + c * Y.col(i + 1);
        }
    }
};



///
/// Perform the QR decomposition of a tridiagonal matrix, a special
/// case of upper Hessenberg matrices.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float`, `double` and `long double`.
///
template <typename Scalar = double>
class TridiagQR: public UpperHessenbergQR<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    typedef typename Matrix::Index Index;

public:
    ///
    /// Default constructor. Computation can
    /// be performed later by calling the compute() method.
    ///
    TridiagQR() :
        UpperHessenbergQR<Scalar>()
    {}

    ///
    /// Constructor to create an object that performs and stores the
    /// QR decomposition of a tridiagonal matrix `mat`.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the major- and sub- diagonal parts of
    /// the matrix are used.
    ///
    TridiagQR(ConstGenericMatrix& mat) :
        UpperHessenbergQR<Scalar>()
    {
        this->compute(mat);
    }

    ///
    /// Conduct the QR factorization of a tridiagonal matrix.
    ///
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    /// Only the major- and sub- diagonal parts of
    /// the matrix are used.
    ///
    void compute(ConstGenericMatrix& mat)
    {
        this->m_n = mat.rows();
        if(this->m_n != mat.cols())
            throw std::invalid_argument("TridiagQR: matrix must be square");

        this->m_mat_T.resize(this->m_n, this->m_n);
        this->m_rot_cos.resize(this->m_n - 1);
        this->m_rot_sin.resize(this->m_n - 1);

        this->m_mat_T.setZero();
        this->m_mat_T.diagonal().noalias() = mat.diagonal();
        this->m_mat_T.diagonal(1).noalias() = mat.diagonal(-1);
        this->m_mat_T.diagonal(-1).noalias() = mat.diagonal(-1);

        // A number of pointers to avoid repeated address calculation
        Scalar *Tii = this->m_mat_T.data(),  // pointer to T[i, i]
               *ptr,                         // some location relative to Tii
               *c = this->m_rot_cos.data(),  // pointer to the cosine vector
               *s = this->m_rot_sin.data(),  // pointer to the sine vector
               r, tmp;
        const Index n2 = this->m_n - 2;
        for(Index i = 0; i < n2; i++)
        {
            // Tii[0] == T[i, i]
            // Tii[1] == T[i + 1, i]
            // r = sqrt(T[i, i]^2 + T[i + 1, i]^2)
            // c = T[i, i] / r, s = -T[i + 1, i] / r
            this->compute_rotation(Tii[0], Tii[1], r, *c, *s);

            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(i + 2)] = G' * T[i:(i + 1), i:(i + 2)]

            // Update T[i, i] and T[i + 1, i]
            // The updated value of T[i, i] is known to be r
            // The updated value of T[i + 1, i] is known to be 0
            Tii[0] = r;
            Tii[1] = Scalar(0);
            // Update T[i, i + 1] and T[i + 1, i + 1]
            // ptr[0] == T[i, i + 1]
            // ptr[1] == T[i + 1, i + 1]
            ptr = Tii + this->m_n;
            tmp = *ptr;
            ptr[0] = (*c) * tmp - (*s) * ptr[1];
            ptr[1] = (*s) * tmp + (*c) * ptr[1];
            // Update T[i, i + 2] and T[i + 1, i + 2]
            // ptr[0] == T[i, i + 2] == 0
            // ptr[1] == T[i + 1, i + 2]
            ptr += this->m_n;
            ptr[0] = -(*s) * ptr[1];
            ptr[1] *= (*c);

            // Move from T[i, i] to T[i + 1, i + 1]
            Tii += this->m_n + 1;
            // Increase c and s by 1
            c++;
            s++;


            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update T[i + 1, (i + 1):(i + 2)]
            // this->m_mat_T(i + 1, i + 1) = (*c) * this->mat_T(i + 1, i + 1) + (*s) * this->m_mat_T(i, i + 1);
            // this->m_mat_T(i + 1, i + 2) *= (*c);
        }
        // For i = n - 2
        this->compute_rotation(Tii[0], Tii[1], r, *c, *s);
        Tii[0] = r;
        Tii[1] = 0;
        ptr = Tii + this->m_n;  // points to T[i - 2, i - 1]
        tmp = *ptr;
        ptr[0] = (*c) * tmp - (*s) * ptr[1];
        ptr[1] = (*s) * tmp + (*c) * ptr[1];

        this->m_computed = true;
    }

    ///
    /// Return the \f$R\f$ matrix in the QR decomposition, which is an
    /// upper triangular matrix.
    ///
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Matrix matrix_R() const
    {
        if(!this->m_computed)
            throw std::logic_error("TridiagQR: need to call compute() first");

        return this->m_mat_T;
    }

    ///
    /// Return the \f$RQ\f$ matrix, the multiplication of \f$R\f$ and \f$Q\f$,
    /// which is a tridiagonal matrix.
    ///
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Matrix matrix_RQ() const
    {
        if(!this->m_computed)
            throw std::logic_error("TridiagQR: need to call compute() first");

        // Make a copy of the R matrix
        Matrix RQ(this->m_n, this->m_n);
        RQ.setZero();
        RQ.diagonal().noalias() = this->m_mat_T.diagonal();
        RQ.diagonal(1).noalias() = this->m_mat_T.diagonal(1);

        // [m11  m12] will point to RQ[i:(i+1), i:(i+1)]
        // [m21  m22]
        Scalar *m11 = RQ.data(), *m12, *m21, *m22;
        const Scalar *c = this->m_rot_cos.data(),
                     *s = this->m_rot_sin.data();
        for(Index i = 0; i < this->m_n - 1; i++)
        {
            m21 = m11 + 1;
            m12 = m11 + this->m_n;
            m22 = m12 + 1;
            const Scalar tmp = *m21;

            // Update diagonal and the below-subdiagonal
            *m11 = (*c) * (*m11) - (*s) * (*m12);
            *m21 = (*c) * tmp - (*s) * (*m22);
            *m22 = (*s) * tmp + (*c) * (*m22);

            // Move m11 to RQ[i+1, i+1]
            m11 = m22;
            c++;
            s++;
        }

        // Copy the below-subdiagonal to above-subdiagonal
        RQ.diagonal(1) = RQ.diagonal(-1);

        return RQ;
    }
};


} // namespace Spectra

#endif // UPPER_HESSENBERG_QR_H
