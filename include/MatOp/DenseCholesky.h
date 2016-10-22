// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DENSE_CHOLESKY_H
#define DENSE_CHOLESKY_H

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <stdexcept>

namespace Spectra {


///
/// \ingroup MatOp
///
template <typename Scalar>
class DenseCholesky
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    const int m_n;
    Eigen::LLT<Matrix> m_solver;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Eigen** matrix object, whose type can be
    /// `Eigen::Matrix<Scalar, ...>` (e.g. `Eigen::MatrixXd` and
    /// `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    ///
    DenseCholesky(ConstGenericMatrix& mat_) :
        m_n(mat_.rows())
    {
        if(mat_.rows() != mat_.cols())
            throw std::invalid_argument("DenseCholesky: matrix must be square");

        m_solver.compute(mat_);
    }

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    int rows() const { return m_n; }
    ///
    /// Return the number of columns of the underlying matrix.
    ///
    int cols() const { return m_n; }

    void lower_triangular_solve(Scalar* x_in, Scalar* y_out) const
    {
        MapVec x(x_in,  m_n);
        MapVec y(y_out, m_n);
        y.noalias() = m_solver.matrixL().solve(x);
    }

    void upper_triangular_solve(Scalar* x_in, Scalar* y_out) const
    {
        MapVec x(x_in,  m_n);
        MapVec y(y_out, m_n);
        y.noalias() = m_solver.matrixU().solve(x);
    }
};


} // namespace Spectra

#endif // DENSE_CHOLESKY_H
