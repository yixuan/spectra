// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_JD_SYM_EIGS_BASE_H
#define SPECTRA_JD_SYM_EIGS_BASE_H

#include <Eigen/Core>
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
        m_op(op),
        m_n(op.rows()),
        m_nev(nev),
        m_max_search_space_size(10*m_nev),
        m_initial_search_space_size(2*m_nev)
    {
        check_argument();
    }

    // If op is an rvalue
    JDSymEigsBase(OpType&& op, Index nev) :
        m_op_container(create_op_container(std::move(op))),
        m_op(m_op_container.front()),
        m_n(m_op.rows()),
        m_nev(nev),
        m_max_search_space_size(10*m_nev),
        m_initial_search_space_size(2*m_nev)
    {
        check_argument();
    }

    ///
    /// Sets the Maxmium SearchspaceSize after which is deflated
    ///
    void setMaxSearchspaceSize(Index max_search_space_size){
        m_max_search_space_size=max_search_space_size;
    }


    ///
    /// Sets the Initial SearchspaceSize for Ritz values
    ///
    void setInitialSearchspaceSize(Index initial_search_space_size){
        m_initial_search_space_size=initial_search_space_size;
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
    Index num_iterations() const { return m_niter; }

    ///
    /// Returns the number of matrix operations used in the computation.
    ///
    Index num_operations() const { return m_nmatop; }

    struct RitzEigenPair
    {
        Vector lambda;  // eigenvalues
        Matrix q;       // Ritz (or harmonic Ritz) eigenvectors
        Matrix U;       // eigenvectors of the small subspace
        Matrix res;     // residues of the pairs
        Array res_norm() const
        {
            return res.colwise().norm();
        }  // norm of the residues
    };

    struct ProjectedSpace
    {
        Matrix vect;   // basis of vectors
        Matrix m_vect;  // A * V
        Matrix vect_m_vect;   // V.T * A * V
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
        m_nmatop = 0;
        m_niter = 0;
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
    std::vector<OpType> m_op_container;
    OpType& m_op;  // object to conduct marix operation,
                   // e.g. matrix-vector product

    Index m_niter = 0;
    const Index m_n;             // dimension of matrix A
    const Index m_nev;           // number of eigenvalues requested
    Index m_max_search_space_size;
    Index m_initial_search_space_size;
    Index m_nmatop = 0;          // number of matrix operations called
    RitzEigenPair m_ritz_pairs;  // Ritz eigen pair structure
    ProjectedSpace m_proj_space; // Projected Space structure     

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
        if (m_nev < 1 || m_nev > m_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");
    }

    void restart(Index size) {
        m_proj_space.vect = m_ritz_pairs.q.leftCols(size);
        m_proj_space.m_vect= m_proj_space.m_vect * m_ritz_pairs.U.leftCols(size); // ? 
        m_proj_space.vect_m_vect = m_proj_space.vect.transpose() * m_proj_space.m_vect;
    }

    void full_update_projected_space() 
    {
        m_proj_space.m_vect = m_op * m_proj_space.vect;
        m_nmatop ++;

        m_proj_space.vect_m_vect = m_proj_space.vect.transpose() * m_proj_space.m_vect;        
        
    }

    void update_projected_space()
    {
      Index old_dim = m_proj_space.vect_m_vect.cols();
      Index new_dim = m_proj_space.vect.cols();
      Index nvec = new_dim - old_dim;

      m_proj_space.m_vect.conservativeResize(Eigen::NoChange, new_dim);
      
      m_proj_space.m_vect.rightCols(nvec) = m_op * m_proj_space.vect.rightCols(nvec);
      m_nmatop ++;

      Matrix tmp_proj = m_proj_space.vect.transpose() * m_proj_space.m_vect.rightCols(nvec);
      m_proj_space.vect_m_vect.conservativeResize(new_dim, new_dim);
      m_proj_space.vect_m_vect.rightCols(nvec) = tmp_proj;
      m_proj_space.vect_m_vect.bottomLeftCorner(nvec, old_dim) =
        m_proj_space.vect_m_vect.topRightCorner(old_dim, nvec).transpose();
    }

public:
    Index compute(SortRule selection = SortRule::LargestMagn, Index maxit = 1000,
                  Scalar tol = 1e-10)
    {
        
        for(m_niter=0; m_niter < maxit; m_niter++)
        {
            bool do_restart = (m_proj_space.current_size() > m_max_search_space_size);
            
            if (do_restart) {
                restart(m_initial_search_space_size);
            }

            update_projected_space();
            


        }
    }
};

}  // namespace Spectra
#endif  // SPECTRA_JD_SYM_EIGS_BASE_H