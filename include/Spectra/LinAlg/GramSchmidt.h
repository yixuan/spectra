// Copyright (C) 2020 Netherlands eScience Center <f.zapata@esciencecenter.nl>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_GRAM_SCHIMDT_H
#define SPECTRA_GRAM_SCHIMDT_H

#include <Eigen/Core>

namespace Spectra {

// Gram-Schmidt process to orthoganlize a given basis A. Each column correspond
// to a vector on the basis.
// The start index indicates from what vector to start
// the orthogonalization, 0 is the default.
// Warnings: Starting the normalization at index n imples that the previous
// n-1 vectors are already orthonormal.

template <typename Scalar>
class Gramschmidt
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Matrix& basis;
    Index start_index = 0;

public:
    Gramschmidt(const Matrix& A, Index nstart) :
        basis{A}, start_index{nstart} {}

    Gramschmidt(const Matrix& A) :
        basis{A} {}

    Matrix orthonormalize()
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
            // two consecutive Gram-schmidt iterations are enough
            // http://stoppels.blog/posts/orthogonalization-performance
            Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() * Q.col(j));
            if (Q.col(j).norm() <= 1E-12 * basis.col(j).norm())
            {
                throw std::runtime_error(
                    "There is a Linear dependencies in Gram-Schmidt."
                    "Hint: try the modified Gram-Schmidt or the QR method.");
            }
            Q.col(j).normalize();
        }
        return Q;
    }
};

}  // namespace Spectra

#endif  //SPECTRA_GRAM_SCHIMDT_H