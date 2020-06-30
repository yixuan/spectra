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
    // If op is an lvalue
    JDSymEigsBase(OpType& op, Index nev) :
        matrix_operator_(op),
        operator_dimension_(op.rows()),
        number_eigenvalues_(nev),
        max_search_space_size_(10*number_eigenvalues_),
        initial_search_space_size_(2*number_eigenvalues_)
    {
        check_argument();
    }

    // If op is an rvalue
    JDSymEigsBase(OpType&& op, Index nev) :
        matrix_op_container_(create_op_container(std::move(op))),
        matrix_operator_(matrix_op_container_.front()),
        operator_dimension_(matrix_operator_.rows()),
        number_eigenvalues_(nev),
        max_search_space_size_(10*number_eigenvalues_),
        initial_search_space_size_(2*number_eigenvalues_)
    {
        check_argument();
    }

    ///
    /// Sets the Maxmium SearchspaceSize after which is deflated
    ///
    void setMaxSearchspaceSize(Index max_search_space_size){
        max_search_space_size_=max_search_space_size;
    }


    ///
    /// Sets the Initial SearchspaceSize for Ritz values
    ///
    void setInitialSearchspaceSize(Index initial_search_space_size){
        initial_search_space_size_=initial_search_space_size;
    }

    ///
    /// Virtual destructor
    ///
    virtual ~JDSymEigsBase() {}

    ///
    /// Returns the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
    CompInfo info() const { return m_info; }

    ///
    /// Returns the number of iterations used in the computation.
    ///
    Index num_iterations() const { return niter_; }

    ///
    /// Returns the number of matrix operations used in the computation.
    ///
    Index num_operations() const { return number_matrix_mul_; }

    struct RitzEigenPairs
    {
        Vector eval;    // eigenvalues
        Matrix evect;   // Ritz (or harmonic Ritz) eigenvectors
        Matrix res;     // residues of the pairs
        Array res_norm() const
        {
            return res.colwise().norm();
        }  // norm of the residues

        Index size() const{return eval.size();}
    };

    struct ProjectedSpace
    {
        Matrix vect;   // basis of vectors
        Matrix m_vect;  // A * V
        
        Index current_size() const
        {
            return vect.cols();
        };                                 // size of the projection i.e. number of cols in V
        Index size_update;                 // size update ...
        BoolArray root_converged;  // keep track of which root have onverged
    };

    ///
    /// Initializes the solver by providing an initial search space.
    ///
    /// \param init_resid Matrix containing initial search space.
    ///
    ///
    void init(const Matrix& init_space)
    {
        number_matrix_mul_ = 0;
        niter_ = 0;
    }

    ///
    /// Initializes the solver by providing  initial search sapce.
    ///
/// Depending n the method this search space is initialized in a different way
    ///
    void init()
    {
        Matrix intial_space = SetupInitialSearchSpace();
        init(intial_space);
    }

protected:
    virtual Matrix SetupInitialSearchSpace() const = 0;

    virtual Matrix CalculateCorrectionVector() const = 0;
    std::vector<OpType> matrix_op_container_;
    OpType& matrix_operator_;  // object to conduct matrix operation,
                   // e.g. matrix-vector product

    Index niter_ = 0;
    const Index operator_dimension_;             // dimension of matrix A
    const Index number_eigenvalues_;           // number of eigenvalues requested
    Index max_search_space_size_;
    Index initial_search_space_size_;
    Index number_matrix_mul_ = 0;          // number of matrix operations called
    RitzEigenPairs ritz_pairs_;  // Ritz eigen pair structure
    ProjectedSpace proj_space_; // Projected Space structure     

private:
    CompInfo m_info = CompInfo::NotComputed;  // status of the computation

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

    void restart(Index size) {
        proj_space_.vect = ritz_pairs_.q.leftCols(size);
        proj_space_.m_vect= proj_space_.m_vect * ritz_pairs_.U.leftCols(size); // ? 
    }

    void full_update_projected_space() 
    {
        proj_space_.m_vect = matrix_operator_ * proj_space_.vect;
        number_matrix_mul_ ++;        
    }

    void update_projected_space()
    {
      Index old_dim = proj_space_.vect_m_vect.cols();
      Index new_dim = proj_space_.vect.cols();
      Index nvec = new_dim - old_dim;

      proj_space_.m_vect.conservativeResize(Eigen::NoChange, new_dim);      
      proj_space_.m_vect.rightCols(nvec) = matrix_operator_ * proj_space_.vect.rightCols(nvec);
      number_matrix_mul_ ++;

    }

    void compute_ritz_pairs() {
        
        // form the small eigenvalue
        Matrix vect_m_vect = proj_space_.vect.transpose() * proj_space_.vect;

        // small eigenvalue problem
        Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(vect_m_vect);
        ritz_pairs_.eval = eigen_solver.eigenvalues();

        // ritz vectors
        ritz_pairs_.evect = proj_space_.vect * eigen_solver.eigenvectors();;

        // residues
        ritz_pairs_.res = proj_space_.m_vect * eigen_solver.eigenvectors(); - ritz_pairs_.q * ritz_pairs_.lambda.asDiagaonal();
    }

    static void sort_ritz_pairs(SortRule selection,RitzEigenPairs & pairs ){
        std::vector<Index> ind=argsort(selection,pairs.lambda);

        RitzEigenPairs temp=pairs;
        // Copy the Ritz values and vectors to m_ritz_val and m_ritz_vec, respectively
        for (Index i = 0; i < pairs.size(); i++)
        {
            pairs.evals[i]=temp.evals[ind[i]];
            pairs.evect.col(i)=temp.evect.col(ind[i]);
            pairs.res.col(i)=temp.res.col(ind[i]);
        }
    }

    Index extend_projected_space() {

        Matrix corr_vect = CalculateCorrectionVector();
        Index num_update = corr_vect.cols();
        proj_space_.vect.conservativeResize(Eigen::NoChange, proj_space_.vect.cols() + num_update);
        proj_space_.vect.rightCols(corr_vect.cols()) = corr_vect;
        return num_update;
    }

    bool check_convergence(Scalar tol) const{
        
        const Array& res_norm = ritz_pairs_.res_norm();
        bool converged = true;
        
        for (Index j = 0; j < proj_space_.size_update; j++) {
            proj_space_.root_converged[j] = (res_norm[j] < tol);
            if (j < number_eigenvalues_) {
            converged &= (res_norm[j] < tol);
            }
        }
        return converged;   
    }

public:
    Index compute(SortRule selection = SortRule::LargestMagn, Index maxit = 1000,
                  Scalar tol = 1e-10)
    {
        
        for(niter_=0; niter_ < maxit; niter_++)
        {
            bool do_restart = (proj_space_.current_size() > max_search_space_size_);
            
            if (do_restart) {
                restart(initial_search_space_size_);
            }

            update_projected_space();
            
            compute_ritz_pairs();
            
            sort_ritz_pairs(selection,ritz_pairs_);

            bool converged = check_convergence(tol);

            Index num_update = extend_projected_space();
            
            //orthogonalize(m_proj_space.vect, num_update);


        }
    }
};

}  // namespace Spectra
#endif  // SPECTRA_JD_SYM_EIGS_BASE_H