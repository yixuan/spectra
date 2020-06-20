// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_SYM_GEIGS_SHIFT_SOLVER_H
#define SPECTRA_SYM_GEIGS_SHIFT_SOLVER_H

#include "SymEigsBase.h"
#include "Util/GEigsMode.h"
#include "MatOp/internal/SymGEigsShiftInvertOp.h"

namespace Spectra {

///
/// \ingroup GEigenSolver
///
/// This class implements the generalized eigen solver for real symmetric
/// matrices, i.e., to solve \f$Ax=\lambda Bx\f$ where \f$A\f$ is symmetric and
/// \f$B\f$ is positive definite. A spectral transform is applied to seek interior
/// generalized eigenvalues with respect to some shift \f$\sigma\f$.
///
/// There are different modes of this solver, specified by the template parameter `Mode`.
/// See the pages for the specialized classes for details.
/// - The shift-and-invert mode transforms the problem to \f$(A-\sigma B)^{-1}Bx=\nu x\f$,
///   where \f$\nu=1/(\lambda-\sigma)\f$.
///   See \ref SymGEigsShiftSolver<Scalar, OpType, BOpType, GEigsMode::ShiftInvert>
///   "SymGEigsShiftSolver (Shift-and-invert mode)" for more details.

// Empty class template
template <typename Scalar,
          typename OpType,
          typename BOpType,
          GEigsMode Mode>
class SymGEigsShiftSolver
{};

///
/// \ingroup GEigenSolver
///
/// This class implements the generalized eigen solver for real symmetric
/// matrices using the shift-and-invert spectral transformation. The original problem is
/// to solve \f$Ax=\lambda Bx\f$, where \f$A\f$ is symmetric and \f$B\f$ is positive definite.
/// The transformed problem is \f$(A-\sigma B)^{-1}Bx=\nu x\f$, where
/// \f$\nu=1/(\lambda-\sigma)\f$, and \f$\sigma\f$ is a user-specified shift.
///

// Partial specialization for mode = GEigsMode::ShiftInvert
template <typename Scalar,
          typename OpType,
          typename BOpType>
class SymGEigsShiftSolver<Scalar, OpType, BOpType, GEigsMode::ShiftInvert> :
    public SymEigsBase<Scalar, SymGEigsShiftInvertOp<Scalar, OpType, BOpType>, BOpType>
{
private:
    using Index = Eigen::Index;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

    using Base = SymEigsBase<Scalar, SymGEigsShiftInvertOp<Scalar, OpType, BOpType>, BOpType>;
    using Base::m_nev;
    using Base::m_ritz_val;

    const Scalar m_sigma;

    // First transform back the Ritz values, and then sort
    void sort_ritzpair(SortRule sort_rule) override
    {
        // The eigenvalues we get from the iteration is nu = 1 / (lambda - sigma)
        // So the eigenvalues of the original problem is lambda = 1 / nu + sigma
        m_ritz_val.head(m_nev).array() = Scalar(1) / m_ritz_val.head(m_nev).array() + m_sigma;
        Base::sort_ritzpair(sort_rule);
    }

public:
    ///
    /// Constructor to create a solver object.
    ///
    SymGEigsShiftSolver(OpType& op, BOpType& Bop, Index nev, Index ncv, const Scalar& sigma) :
        Base(SymGEigsShiftInvertOp<Scalar, OpType, BOpType>(op, Bop), Bop, nev, ncv),
        m_sigma(sigma)
    {
        op.set_shift(m_sigma);
    }
};

}  // namespace Spectra

#endif  // SPECTRA_SYM_GEIGS_SHIFT_SOLVER_H
