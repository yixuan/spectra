

// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_RITZ_PAIR_H
#define SPECTRA_RITZ_PAIR_H

#include <Eigen/Dense>
#include "SelectionRule.h"

namespace Spectra {
template <typename Scalar>
class SearchSpace;

template <typename Scalar>
class RitzPairs
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;

public:
    RitzPairs() = default;

    void compute_eigen_pairs(const SearchSpace<Scalar>& search_space);

    Index size() const { return values_.size(); }

    void sort(SortRule selection)
    {
        std::vector<Index> ind = argsort(selection, values_);
        RitzPairs<Scalar> temp = *this;
        for (Index i = 0; i < size(); i++)
        {
            values_[i] = temp.values_[ind[i]];
            vectors_.col(i) = temp.vectors_.col(ind[i]);
            residues_.col(i) = temp.residues_.col(ind[i]);
        }
    }

    bool check_convergence(Scalar tol, Index number_eigenvalues) const
    {
        const Array norms = residues_.colwise().norm();
        bool converged = true;
        root_converged_ = BoolArray::Zero(norms.size());
        for (Index j = 0; j < norms.size(); j++)
        {
            root_converged_[j] = (norms[j] < tol);
            if (j < number_eigenvalues)
            {
                converged &= (norms[j] < tol);
            }
        }
        return converged;
    }

    const Matrix& RitzVectors() const { return vectors_; }
    const Vector& RitzValues() const { return values_; }
    const Matrix& SmallRitzVectors() const { return small_vectors_; }
    const Matrix& Residues() const { return residues_; }
    const BoolArray& ConvergedEigenvalues() const { return root_converged_; }

private:
    Vector values_;         // eigenvalues
    Matrix small_vectors_;  // eigenvectors of the small problem, makes restart cheaper.
    Matrix vectors_;        // Ritz (or harmonic Ritz) eigenvectors
    Matrix residues_;       // residues of the pairs
    BoolArray root_converged_;
};

}  // namespace Spectra
#include "SearchSpace.h"
namespace Spectra {

template <typename Scalar>
void RitzPairs<Scalar>::compute_eigen_pairs(const SearchSpace<Scalar>& search_space)
{
    const Matrix& basis_vectors = search_space.BasisVectors();
    const Matrix& op_basis_prod = search_space.OperatorBasisProduct();

    // form the small eigenvalue
    Matrix small_matrix = basis_vectors.transpose() * op_basis_prod;

    // small eigenvalue problem
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(small_matrix);
    values_ = eigen_solver.eigenvalues();
    small_vectors_ = eigen_solver.eigenvectors();

    // ritz vectors
    vectors_ = basis_vectors * small_vectors_;

    // residues
    residues_ = op_basis_prod * small_vectors_ - vectors_ * values_.asDiagaonal();
}

}  // namespace Spectra

#endif  // SPECTRA_RITZ_PAIR_H