// Copyright (C) 2020 Netherlands eScience Center <f.zapata@esciencecenter.nl>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_JD_SYM_EIGS_DPR_H
#define SPECTRA_JD_SYM_EIGS_DPR_H

#include <Eigen/Dense>
#include "JDSymEigsBase.h"

namespace Spectra {

template <typename Scalar, typename OpType>
class JDSymEigsDPR : public JDSymEigsBase<Scalar, OpType>
{
public:
    Matrix SetupInitialSearchSpace() const final
    {
    }

    /// compute the corrections using the DPR method.
    /// \return new correction vectors.
    Matrix CalculateCorrectionVector() const final
    {
        Vector diagonal(operator_dimension_);
        for (Index i = 0; i < operator_dimension_; i++)
        {
            diagonal(i) = matrix_operator_(i, i);
        }
        Index nresidues = search_space_.size();
        const Vector& eigenvalues = ritz_pairs_.RitzValues();
        Matrix correction = Matrix::zero(operator_dimension_, nresidues);
        for (Index k = 0; k < nresidues; k++)
        {
            Vector tmp = Vector::Constant(eigenvalues(k)) - diagonal;
            correction.col(k) = residues / tmp;
        }
        return correction;
    }
};

}  // namespace Spectra

#endif  // SPECTRA_JD_SYM_EIGS_DPR_H