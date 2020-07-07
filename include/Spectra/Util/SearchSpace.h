// Copyright (C) 2020 Netherlands eScience Center <n.renauld@esciencecenter.nl>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_SEARCH_SPACE_H
#define SPECTRA_SEARCH_SPACE_H

#include <Eigen/Dense>
#include "RitzPairs.h"
#include "../LinAlg/Orthogonalization.h"

namespace Spectra {
template <typename Scalar>

///
/// \ingroup Search Space
///
/// This class handles the creation and manipulation of the search space
/// for iterative eigensolvers such as Davidson, Jacobi-Davidson etc ....

class SearchSpace
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

public:
    SearchSpace() = default;

    /// returns the current size of the search space
    Index size() const
    {
        return basis_vectors_.cols();
    }

    /// Updates the matrix formed by the operator applied to the search space
    /// after the addition of new vectors in the search space. Only the product
    /// of the operator with the new vectors is computed and the results is appended
    /// to the op_basis_product member variable
    /// \param OpType operator representing the matrix
    template <typename OpType>
    void update_operator_basis_product(OpType &op)
    {
        Index nvec = basis_vectors_.cols() - op_basis_product_.cols();
        op_basis_product_.conservativeResize(Eigen::NoChange, basis_vectors_.cols());
        op_basis_product_.rightCols(nvec) = op * basis_vectors_.rightCols(nvec);
    }

    void InitializeSearchSpace(const Eigen::Ref<const Matrix> &initial_vectors)
    {
        basis_vectors_ = initial_vectors;
        op_basis_product_ = Matrix(initial_vectors.rows(), 0);
    }

    /// Restart the search space by reducing the basis vector to the last
    /// Ritz eigenvector
    /// \param ritz_pair Instance of a RitzPair class
    /// \param size size of the restart
    void restart(const RitzPairs<Scalar> &ritz_pairs, Index size)
    {
        basis_vectors_ = ritz_pairs.RitzVectors().leftCols(size);
        op_basis_product_ = op_basis_product_ * ritz_pairs.SmallRitzVectors().leftCols(size);
    }

    /// Append new vector to the search space and
    /// orthogonalie the resulting matrix
    /// \param new_vect matrix of new correction vectors
    void extend_basis(const Matrix &new_vect)
    {
        Index leftColstoSkip = size();
        append_new_vectors_to_basis(new_vect);
        Spectra::twice_is_enough_orthogonalisation(basis_vectors_, leftColstoSkip);
    }

    /// Returns the basis vectors
    const Matrix &BasisVectors() const { return basis_vectors_; }

    /// Returns the operator applied to basis vector
    const Matrix &OperatorBasisProduct() const { return op_basis_product_; }

private:
    Matrix basis_vectors_;
    Matrix op_basis_product_;

    /// Append new vector to the basis
    /// \param new_vect matrix of new correction vectors
    void append_new_vectors_to_basis(const Matrix &new_vect)
    {
        Index num_update = new_vect.cols();
        basis_vectors_.conservativeResize(Eigen::NoChange, basis_vectors_.cols() + num_update);
        basis_vectors_.rightCols(new_vect.cols()) = new_vect;
    }
};

}  // namespace Spectra

#endif  // SPECTRA_SEARCH_SPACE_H