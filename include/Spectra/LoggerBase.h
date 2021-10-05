
// Copyright (C) 2018-2021 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_LOGGER_BASE_H
#define SPECTRA_LOGGER_BASE_H

#include <Eigen/Core>
#include <vector>  // std::vector

#include "Util/Version.h"
#include "Util/TypeTraits.h"

namespace Spectra {

template <typename Scalar, typename Vector>
class LoggerBase
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

public:
    LoggerBase() {}

    ///
    /// Virtual destructor
    ///
    virtual ~LoggerBase() {}

    // I am not sure what should be in the Data, probably iteration count, residues, current eigenvalues, for davidson maybe subspace size, number_of_converged eigenvalues().

    ///
    /// Virtual logging function
    ///
    virtual void iteration_log(const Index& iteration, const Index& number_of_converged, const Index& subspace_size, const Vector& current_eigenvalues, const Vector& residues, const BoolArray& current_eig_converged);
};

}  // namespace Spectra

#endif  // SPECTRA_LOGGER_BASE_H
