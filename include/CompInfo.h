// Copyright (C) 2015 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef COMP_INFO_H
#define COMP_INFO_H

namespace Spectra {


///
/// \ingroup Enumerations
///
/// The enumeration to report the status of computation.
///
enum COMPUTATION_INFO
{
    SUCCESSFUL = 0,    ///< Computatoin was successful.

    NOT_COMPUTED,      ///< Computation has not been conducted. Users should call
                       ///< the `compute()` member function of solvers.

    NOT_CONVERGING     ///< Some eigenvalues did not converge. The `compute()`
                       ///< function returns the number of converged eigenvalues.
};


} // namespace Spectra

#endif // COMP_INFO_H
