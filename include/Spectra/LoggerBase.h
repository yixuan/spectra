
// Copyright (C) 2018-2021 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_LOGGER_BASE_H
#define SPECTRA_LOGGER_BASE_H

#include <Eigen/Core>

namespace Spectra {

template <typename Scalar, typename Vector>
class IterationData
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

public:
    const Index& iteration;
    const Index& number_of_converged;
    const Index& subspace_size;
    const Vector& current_eigenvalues;
    const Vector& residues;
    const BoolArray& current_eig_converged;
    IterationData(const Index& it, const Index& num_conv, const Index& sub_size, const Vector& cur_eigv, const Vector& res, const BoolArray& cur_eig_conv) :
        iteration(it),
        number_of_converged(num_conv),
        subspace_size(sub_size),
        current_eigenvalues(cur_eigv),
        residues(res),
        current_eig_converged(cur_eig_conv)
    {}
};

template <typename Scalar, typename Vector>
class LoggerBase
{
public:
    LoggerBase() {}

    ///
    /// Virtual destructor
    ///
    virtual ~LoggerBase() {}

    ///
    /// Virtual logging function
    ///
    virtual void iteration_log(const IterationData<Scalar, Vector>& data);
};

}  // namespace Spectra

#endif  // SPECTRA_LOGGER_BASE_H
