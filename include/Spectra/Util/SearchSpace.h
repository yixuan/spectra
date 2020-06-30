// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_SEARCH_SPACE_H
#define SPECTRA_SEARCH_SPACE_H

#include <Eigen/Dense>
#include "RitzPairs.h"

namespace Spectra {
template <typename Scalar>
class SearchSpace
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;

public:
    SearchSpace() = default;

    Index size() const
    {
        return basis_vectors_.cols();
    }

    template <typename OpType>
    void update_operator_basis_product(OpType &op)
    {
        Index nvec = basis_vectors_.cols() - op_basis_product_.cols();
        op_basis_product_.conservativeResize(Eigen::NoChange, basis_vectors_.cols());
        op_basis_product_.rightCols(nvec) = op * basis_vectors_.rightCols(nvec);
    }

    template <typename OpType>
    void full_update(OpType &op)
    {
        op_basis_product_ = op * basis_vectors_;
    }

    void restart(const RitzPairs<Scalar> &ritz_pairs, Index size)
    {
        basis_vectors_ = ritz_pairs.Vectors().leftCols(size);
        op_basis_product_ = op_basis_product_ * ritz_pairs.SmallRitzVectors().leftCols(size);
    }

    void extend_basis(const Matrix &new_vect)
    {
        Index num_update = new_vect.cols();
        append_new_vectors_to_basis(new_vect);
        //basis_vectors_ = orthogonalize(basis_vectors_, num_update);

        //TODO orthogonalize
    }

    const Matrix &BasisVectors() const { return basis_vectors_; }
    Matrix &BasisVectors() { return basis_vectors_; }
    const Matrix &OperatorBasisProduct() const { return op_basis_product_; }

private:
    Matrix basis_vectors_;
    Matrix op_basis_product_;

    void append_new_vectors_to_basis(const Matrix &new_vect)
    {
        Index num_update = new_vect.cols();
        basis_vectors_.conservativeResize(Eigen::NoChange, basis_vectors_.cols() + num_update);
        basis_vectors_.rightCols(new_vect.cols()) = new_vect;
    }
};

}  // namespace Spectra

#endif  // SPECTRA_SEARCH_SPACE_H