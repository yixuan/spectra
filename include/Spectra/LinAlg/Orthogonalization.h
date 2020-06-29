// Copyright (C) 2020 Netherlands eScience Center <f.zapata@esciencecenter.nl>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_ORTHOGONALIZATION_H
#define SPECTRA_ORTHOGONALIZATION_H

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Spectra {

/** \ingroup LinAlg
  *
  *
  * \class 
  *
  * \brief Methods to orthogalize a given basis A.
  *
  * Each column correspond to a vector on the basis.
  * The start index indicates from what vector to start
  * the orthogonalization, 0 is the default.
  * Warnings: Starting the normalization at index n imples that the previous
  * n-1 vectors are already orthonormal.
  */
template <typename Scalar>
class Orthogonalization
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    const Matrix& basis;
    Index start_index = 0;
    void check_linear_dependency(const Vector& vec, Index j)
    {
        if (vec.norm() <= 1E-12 * basis.col(j).norm())
        {
            throw std::runtime_error(
                "There is a Linear dependencies in Gram-Schmidt."
                "Hint: try the QR method.");
        }
    }

public:
    /// \param mat Matrix type can be `Eigen::Matrix<Scalar, ...>` (e.g.
    /// `Eigen::MatrixXd` and `Eigen::MatrixXf`)
    /// \param nstart index of the column from which to start the process
    Orthogonalization(const Matrix& A, Index nstart) :
        basis{A}, start_index{nstart} {}

    Orthogonalization(const Matrix& A) :
        basis{A} {}

    /// two consecutive Gram-schmidt iterations are enough to converge
    /// http://stoppels.blog/posts/orthogonalization-performance
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    Matrix twice_modified_gramschmidt()
    {
        Matrix Q = basis;
        Index nstart = start_index;
        if (start_index == 0)
        {
            Q.col(0).normalize();
            nstart = 1;
        }
        for (Index j = nstart; j < basis.cols(); ++j)
        {
            Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() * basis.col(j));
            Q.col(j).normalize();
            Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() * Q.col(j));
            check_linear_dependency(Q.col(j), j);
            Q.col(j).normalize();
        }
        return Q;
    }

    /// Standard modified Gram-Schmidt process
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    Matrix modified_gramschmidt()
    {
        Matrix Q = basis;
        Index nstart = start_index;
        if (start_index == 0)
        {
            Q.col(0).normalize();
            nstart = 1;
        }
        for (Index j = nstart; j < basis.cols(); ++j)
        {
            Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() * basis.col(j));
            check_linear_dependency(Q.col(j), j);
            Q.col(j).normalize();
        }
        return Q;
    }

    /// Use the QR method to orthonormalize the basis.
    /// \return Returned matrix type will be `Eigen::Matrix<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    Matrix QR()
    {
        Index nrows = basis.rows();
        Index ncols = basis.cols();
        ncols = std::min(nrows, ncols);
        Matrix I = Matrix::Identity(nrows, ncols);
        Eigen::HouseholderQR<Matrix> qr(basis);
        return qr.householderQ() * I;
    }
};

}  // namespace Spectra

#endif  //SPECTRA_ORTHOGONALIZATION_H