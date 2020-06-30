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
    // virtual Matrix SetupInitialSearchSpace() const final;

    Matrix CalculateCorrectionVector() const final
    {
        Index nresidues = search_space_.size();
        Matrix correction = Matrix::zero(operator_dimension_, nresidues);
        for (Index k = 0; k < ncols; k++)
        {
            // Vector tmp =
            correction.col(k) = residues / tmp;
        }
    }

    // let d = self.target.diagonal();
    // let mut correction = DMatrix::<f64>::zeros(self.target.nrows(), residues.ncols());
    // for (k, lambda) in eigenvalues.iter().enumerate() {
    //     let tmp = DVector::<f64>::repeat(self.target.nrows(), *lambda) - &d;
    //     let rs = residues.column(k).component_div(&tmp);
    //     correction.set_column(k, &rs);
    // }
    correction
}

};  // namespace Spectra

}  // namespace Spectra

#endif  // SPECTRA_JD_SYM_EIGS_DPR_H