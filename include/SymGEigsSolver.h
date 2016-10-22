// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYM_GEIGS_SOLVER_H
#define SYM_GEIGS_SOLVER_H

#include "SymEigsSolver.h"
#include "Util/GEigsMode.h"
#include "MatOp/SymGEigsCholeskyOp.h"


namespace Spectra {


///
/// \defgroup GEigenSolver Generalized Eigen Solvers
///
/// Generalized eigen solvers for different types of problems.
///

///
/// \ingroup GEigenSolver
///
/// This class implements the generalized eigen solver for real symmetric
/// matrices.

// Empty class template
template < typename Scalar,
           int SelectionRule,
           typename OpType,
           typename BOpType,
           int GEigsMode >
class SymGEigsSolver
{};



///
/// \ingroup GEigenSolver
///

// Partial specialization for GEigsMode = GEIGS_CHOLESKY
template < typename Scalar,
           int SelectionRule,
           typename OpType,
           typename BOpType >
class SymGEigsSolver<Scalar, SelectionRule, OpType, BOpType, GEIGS_CHOLESKY>:
    public SymEigsSolver< Scalar, SelectionRule, SymGEigsCholeskyOp<Scalar, OpType, BOpType> >
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    BOpType* m_Bop;

public:
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op_  Pointer to the \f$A\f$ matrix operation object. It
    ///             should implement the matrix-vector multiplication operation of \f$A\f$:
    ///             calculating \f$Ay\f$ for any vector \f$y\f$. Users could either
    ///             create the object from the DenseSymMatProd wrapper class, or
    ///             define their own that impelemnts all the public member functions
    ///             as in DenseSymMatProd.
    /// \param Bop_ Pointer to the \f$B\f$ matrix operation object. It
    ///             represents a Cholesky decomposition of \f$B\f$, and should
    ///             implement the lower and upper triangular solving operations:
    ///             calculating \f$L^{-1}y\f$ and \f$(L')^{-1}y\f$ for any vector
    ///             \f$y\f$, where \f$LL'=B\f$. Users could either
    ///             create the object from the DenseCholesky wrapper class, or
    ///             define their own that impelemnts all the public member functions
    ///             as in DenseCholesky.
    /// \param nev_ Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv_ Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv_` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\cdot nev\f$.
    ///
    SymGEigsSolver(OpType* op_, BOpType* Bop_, int nev_, int ncv_) :
        SymEigsSolver< Scalar, SelectionRule, SymGEigsCholeskyOp<Scalar, OpType, BOpType> >(
            new SymGEigsCholeskyOp<Scalar, OpType, BOpType>(*op_, *Bop_), nev_, ncv_
        ),
        m_Bop(Bop_)
    {}

    ~SymGEigsSolver()
    {
        delete this->m_op;
    }

    Matrix eigenvectors(int nvec)
    {
        Matrix res = SymEigsSolver< Scalar, SelectionRule, SymGEigsCholeskyOp<Scalar, OpType, BOpType> >::eigenvectors(nvec);
        Vector tmp(res.rows());
        const int nconv = res.cols();
        for(int i = 0; i < nconv; i++)
        {
            m_Bop->upper_triangular_solve(&res(0, i), tmp.data());
            res.col(i) = tmp;
        }

        return res;
    }

    Matrix eigenvectors()
    {
        return SymGEigsSolver<Scalar, SelectionRule, OpType, BOpType, GEIGS_CHOLESKY>::eigenvectors(this->m_nev);
    }
};


} // namespace Spectra

#endif // SYM_GEIGS_SOLVER_H
