// Copyright (C) 2020 Netherlands eScience Center <n.renauld@esciencecenter.nl>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_RITZ_PAIRS_H
#define SPECTRA_RITZ_PAIRS_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "../Util/SelectionRule.h"

namespace Spectra {

template <typename Scalar>
class SearchSpace;

/// This class handles the creation and manipulation of Ritz eigen pairs
/// for iterative eigensolvers such as Davidson, Jacobi-Davidson, etc.
template <typename Scalar>
class RitzPairs
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

    Vector values_;         // eigenvalues
    Matrix small_vectors_;  // eigenvectors of the small problem, makes restart cheaper.
    Matrix vectors_;        // Ritz (or harmonic Ritz) eigenvectors
    Matrix residues_;       // residues of the pairs
    BoolArray root_converged_;

public:
    RitzPairs() = default;

    /// Compute the eigen values/vectors
    ///
    /// \param search_space Instance of the class handling the search space
    /// \return Eigen::ComputationalInfo Whether small eigenvalue problem worked
    Eigen::ComputationInfo compute_eigen_pairs(const SearchSpace<Scalar>& search_space);

    /// Returns the size of the ritz eigen pairs
    ///
    /// \return Eigen::Index Number of pairs
    Index size() const { return values_.size(); }

    /// Sort the eigen pairs according to the selection rule
    ///
    /// \param selection Sorting rule
    void sort(SortRule selection)
    {
        std::vector<Index> ind = argsort(selection, values_);
        RitzPairs<Scalar> temp = *this;
        for (Index i = 0; i < size(); i++)
        {
            values_[i] = temp.values_[ind[i]];
            vectors_.col(i) = temp.vectors_.col(ind[i]);
            residues_.col(i) = temp.residues_.col(ind[i]);
            small_vectors_.col(i) = temp.small_vectors_.col(ind[i]);
        }
    }

    /// Checks if the algorithm has converged and updates root_converged
    ///
    /// \param tol Tolerance for convergence
    /// \param number_eigenvalue Number of request eigenvalues
    /// \return bool true if all eigenvalues are converged
    bool check_convergence(Scalar tol, Index number_eigenvalues)
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

    const Matrix& ritz_vectors() const { return vectors_; }
    const Vector& ritz_values() const { return values_; }
    const Matrix& small_ritz_vectors() const { return small_vectors_; }
    const Matrix& residues() const { return residues_; }
    const BoolArray& converged_eigenvalues() const { return root_converged_; }
};

}  // namespace Spectra

#include "SearchSpace.h"

namespace Spectra {

/// Creates the small space matrix and computes its eigen pairs
/// Also computes the ritz vectors and residues
///
/// \param search_space Instance of the SearchSpace class
template <typename Scalar>
Eigen::ComputationInfo RitzPairs<Scalar>::compute_eigen_pairs(const SearchSpace<Scalar>& search_space)
{
    const Matrix& basis_vectors = search_space.basis_vectors();
    const Matrix& op_basis_prod = search_space.operator_basis_product();

    // Form the small eigenvalue
    Matrix small_matrix = basis_vectors.transpose() * op_basis_prod;

    // Small eigenvalue problem
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(small_matrix);
    values_ = eigen_solver.eigenvalues();
    small_vectors_ = eigen_solver.eigenvectors();

    // Ritz vectors
    vectors_ = basis_vectors * small_vectors_;

    // Residues
    residues_ = op_basis_prod * small_vectors_ - vectors_ * values_.asDiagonal();
    return eigen_solver.info();
}

}  // namespace Spectra

#endif  // SPECTRA_RITZ_PAIRS_H
