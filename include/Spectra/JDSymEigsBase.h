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
namespace spectra {

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
        m_nev(nev)
    {
        if (nev < 1 || nev > m_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

        if (ncv <= nev || ncv > m_n)
            throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");
    }

    // If op is an rvalue
    JDSymEigsBase(OpType&& op, Index nev) :
        m_op_container(create_op_container(std::move(op))),
        m_op(m_op_container.front()),
        m_n(m_op.rows()),
        m_nev(nev)

    {
        if (nev < 1 || nev > m_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

        if (ncv <= nev || ncv > m_n)
            throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");
    }

    ///
    /// Virtual destructor
    ///
    virtual ~JDSymEigsBase() {}

protected:
    std::vector<OpType> m_op_container;
    OpType& m_op;  // object to conduct marix operation,
                   // e.g. matrix-vector product

    Index m_niter = 0;
    const Index m_n;    // dimension of matrix A
    const Index m_nev;  // number of eigenvalues requested
    Index         m_nmatop=0;     // number of matrix operations called

private:
    CompInfo m_info = CompInfo::NotComputed;  // status of the computation

    // Move rvalue object to the container
    static std::vector<OpType> create_op_container(OpType&& rval)
    {
        std::vector<OpType> container;
        container.emplace_back(std::move(rval));
        return container;
    }

    void check_argument() const {
        if (nev < 1 || nev > m_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

        if (ncv <= nev || ncv > m_n)
            throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");

    }
}

}  // namespace spectra
