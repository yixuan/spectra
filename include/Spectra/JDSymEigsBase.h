// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_JD_SYM_EIGS_BASE_H
#define SPECTRA_JD_SYM_EIGS_BASE_H

#include <Eigen/Dense>
#include <vector>     // std::vector
#include <cmath>      // std::abs, std::pow
#include <algorithm>  // std::min
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::move

#include "Util/Version.h"
#include "Util/TypeTraits.h"
#include "Util/SelectionRule.h"
#include "Util/CompInfo.h"
#include "Util/SimpleRandom.h"
#include "Util/SearchSpace.h"
#include "Util/RitzPairs.h"

namespace Spectra {

template <typename Scalar,
          typename OpType>
class JDSymEigsBase
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using MapMat = Eigen::Map<Matrix>;
    using MapVec = Eigen::Map<Vector>;
    using MapConstVec = Eigen::Map<const Vector>;

public:
    JDSymEigsBase(OpType& op, Index nev) :
        matrix_operator_(op),
        operator_dimension_(op.rows()),
        number_eigenvalues_(nev),
        max_search_space_size_(10 * number_eigenvalues_),
        initial_search_space_size_(2 * number_eigenvalues_)
    {
        check_argument();
    }

    ///
    /// Sets the Maxmium SearchspaceSize after which is deflated
    ///
    void setMaxSearchspaceSize(Index max_search_space_size)
    {
        max_search_space_size_ = max_search_space_size;
    }

    ///
    /// Sets the Initial SearchspaceSize for Ritz values
    ///
    void setInitialSearchspaceSize(Index initial_search_space_size)
    {
        initial_search_space_size_ = initial_search_space_size;
    }

    ///
    /// Virtual destructor
    ///
    virtual ~JDSymEigsBase() {}

    ///
    /// Returns the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
    CompInfo info() const { return info_; }

    ///
    /// Returns the number of iterations used in the computation.
    ///
    Index num_iterations() const { return niter_; }

    Vector eigenvalues() const { return ritz_pairs_.RitzValues().head(number_eigenvalues_); }

    Matrix eigenvectors() const { return ritz_pairs_.RitzVectors().leftCols(number_eigenvalues_); }

    ///
    /// Initializes the solver by providing an initial search space.
    ///
    /// \param init_resid Matrix containing initial search space.
    ///
    ///
    void init(const Eigen::Ref<const Matrix>& init_space)
    {
        niter_ = 0;
        search_space_.BasisVectors() = init_space;
    }

    ///
    /// Initializes the solver by providing  initial search sapce.
    ///
    /// Depending n the method this search space is initialized in a different way
    ///
    void init()
    {
        Matrix intial_space = SetupInitialSearchSpace();
          //TODO orthogonalize
        init(intial_space);
    }

protected:
    virtual Matrix SetupInitialSearchSpace() const = 0;

    virtual Matrix CalculateCorrectionVector() const = 0;
    const OpType& matrix_operator_;  // object to conduct marix operation,
                                     // e.g. matrix-vector product

    Index niter_ = 0;
    const Index operator_dimension_;  // dimension of matrix A
    const Index number_eigenvalues_;  // number of eigenvalues requested
    Index max_search_space_size_;
    Index initial_search_space_size_;
    RitzPairs<Scalar> ritz_pairs_;    // Ritz eigen pair structure
    SearchSpace<Scalar> search_space_;    // search space

private:
    CompInfo info_ = CompInfo::NotComputed;  // status of the computation

    // Move rvalue object to the container
    static std::vector<OpType> create_op_container(OpType&& rval)
    {
        std::vector<OpType> container;
        container.emplace_back(std::move(rval));
        return container;
    }

    void check_argument() const
    {
        if (number_eigenvalues_ < 1 || number_eigenvalues_ > operator_dimension_ - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");
    }

public:
    Index compute(SortRule selection = SortRule::LargestMagn, Index maxit = 1000,
                  Scalar tol = 1e-10)
    {
        for (niter_ = 0; niter_ < maxit; niter_++)
        {
            bool do_restart = (search_space_.size() > max_search_space_size_);

            if (do_restart)
            {
                search_space_.restart(ritz_pairs_, initial_search_space_size_);
            }

            search_space_.update_operator_basis_product(matrix_operator_);

            ritz_pairs_.compute_eigen_pairs(search_space_);

            ritz_pairs_.sort(selection);

            bool converged = ritz_pairs_.check_convergence(tol,number_eigenvalues_);
            if(converged){
                info_=CompInfo::Successful;
                break;
            }else if(niter_ == maxit-1){
                info_=CompInfo::NotConverging;
                break;
            }

            Matrix corr_vect = CalculateCorrectionVector();
            
            search_space_.extend_basis(corr_vect);
        }
        return Index(ritz_pairs_.ConvergedEigenvalues().sum());
    }
};

}  // namespace Spectra
#endif  // SPECTRA_JD_SYM_EIGS_BASE_H