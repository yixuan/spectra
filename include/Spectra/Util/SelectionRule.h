// Copyright (C) 2016-2022 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_SELECTION_RULE_H
#define SPECTRA_SELECTION_RULE_H

#include <vector>     // std::vector
#include <cmath>      // std::abs
#include <algorithm>  // std::sort
#include <complex>    // std::complex
#include <utility>    // std::pair
#include <stdexcept>  // std::invalid_argument

#include <Eigen/Core>
#include "TypeTraits.h"

namespace Spectra {

///
/// \defgroup Enumerations Enumerations
///
/// Enumeration types for the selection rule of eigenvalues.
///

///
/// \ingroup Enumerations
///
/// The enumeration of selection rules of desired eigenvalues.
///

enum class SortRule
{
    LargestMagn,  ///< Select eigenvalues with largest magnitude. Magnitude
                  ///< means the absolute value for real numbers and norm for
                  ///< complex numbers. Applies to both symmetric and general
                  ///< eigen solvers.

    LargestReal,  ///< Select eigenvalues with largest real part. Only for general eigen solvers.

    LargestImag,  ///< Select eigenvalues with largest imaginary part (in magnitude). Only for general eigen solvers.

    LargestAlge,  ///< Select eigenvalues with largest algebraic value, considering
                  ///< any negative sign. Only for symmetric eigen solvers.

    SmallestMagn,  ///< Select eigenvalues with smallest magnitude. Applies to both symmetric and general
                   ///< eigen solvers.

    SmallestReal,  ///< Select eigenvalues with smallest real part. Only for general eigen solvers.

    SmallestImag,  ///< Select eigenvalues with smallest imaginary part (in magnitude). Only for general eigen solvers.

    SmallestAlge,  ///< Select eigenvalues with smallest algebraic value. Only for symmetric eigen solvers.

    BothEnds  ///< Select eigenvalues half from each end of the spectrum. When
              ///< `nev` is odd, compute more from the high end. Only for symmetric eigen solvers.

};

///
/// Eigenvalue solvers use this class to determine which eigenvalues to select.
/// Values of the type SortRule are automatically cast to an appropriate EigenvalueSorter.
///
/// You can also provide your own implementation to determine which eigenvalues to select.
/// For symmetric solvers, this should have a real scalar type.
/// \code C++
/// SymEigsSolver<...> eigsSolver(...);
/// eigsSolver.init();
/// EigenvalueSorter<double> closestToTarget{[](double eigenvalue) { return std::abs(eigenvalue - 10.); }};
/// eigsSolver.compute(closestToTarget);
/// \endcode
/// For general solvers, the scalar type should be complex.
/// \code C++
/// GenEigsSolver<...> eigsSolver(...);
/// eigsSolver.init();
/// EigenvalueSorter<std::complex<double>> closestToTarget{[](std::complex<double> eigenvalue) { return std::abs(eigenvalue - 10.); }};
/// eigsSolver.compute(closestToTarget);
/// \endcode

template <typename Scalar>
class EigenvalueSorter
{
private:
    bool m_both_ends;
    std::function<ElemType<Scalar>(Scalar)> m_target;

public:
    using Index = Eigen::Index;
    using IndexArray = std::vector<Index>;

    ///
    /// \param target Eigenvalues will be sorted according to this target. Only the values with the lowest target will be computed.
    /// \param both_ends If both_ends is true, half of the eigenvalues with the lowest target and half of the eigenvalues with the highest target will be computed.
    EigenvalueSorter(std::function<ElemType<Scalar>(Scalar)> target, bool both_ends) :
        m_both_ends(both_ends), m_target(target)
    {
    }

    ///
    /// \param target Eigenvalues will be sorted according to this target. Only the values with the lowest target will be computed.
    explicit EigenvalueSorter(std::function<ElemType<Scalar>(Scalar)> target) :
        m_both_ends(false), m_target(target)
    {
    }

    /// This constructor casts a SortRule to an appropriate EigenvalueSorter, for complex scalar types
    template <class T = Scalar>
    EigenvalueSorter(SortRule rule, typename std::enable_if<Eigen::NumTraits<T>::IsComplex>::type* = nullptr)
    {
        // The scalar-type is complex

        m_both_ends = false;
        if (rule == SortRule::LargestMagn)
            m_target = [](Scalar x) { using std::abs; return -std::abs(x); };
        else if (rule == SortRule::LargestReal)
            m_target = [](Scalar x) {
                return -x.real();
            };
        else if (rule == SortRule::LargestImag)
            m_target = [](Scalar x) {
                using std::abs;
                return -abs(x.imag());
            };
        else if (rule == SortRule::SmallestMagn)
            m_target = [](Scalar x) { using std::abs; return -std::abs(x); };
        else if (rule == SortRule::SmallestReal)
            m_target = [](Scalar x) {
                return x.real();
            };
        else if (rule == SortRule::SmallestImag)
            m_target = [](Scalar x) {
                using std::abs;
                return abs(x.imag());
            };
        else
            throw std::invalid_argument("unsupported selection rule for complex types");
    }

    /// This constructor casts a SortRule to an appropriate EigenvalueSorter, for real scalar types
    template <class T = Scalar>
    EigenvalueSorter(SortRule rule, typename std::enable_if<!Eigen::NumTraits<T>::IsComplex>::type* = nullptr)
    {
        // The scalar-type is real

        m_both_ends = rule == SortRule::BothEnds;
        if (rule == SortRule::LargestMagn)
            m_target = [](Scalar x) { using std::abs; return -std::abs(x); };
        else if (rule == SortRule::LargestReal)
            m_target = [](Scalar x) {
                return -x;
            };
        else if (rule == SortRule::LargestAlge || rule == SortRule::BothEnds)
            m_target = [](Scalar x) {
                return -x;
            };
        else if (rule == SortRule::SmallestMagn)
            m_target = [](Scalar x) { using std::abs; return -std::abs(x); };
        else if (rule == SortRule::SmallestReal)
            m_target = [](Scalar x) {
                return x;
            };
        else if (rule == SortRule::SmallestAlge)
            m_target = [](Scalar x) {
                return x;
            };
        else
            throw std::invalid_argument("unsupported selection rule for real types");
    }

    IndexArray argsort(const Scalar* data, Index size) const
    {
        IndexArray index;
        index.resize(size);
        for (Index i = 0; i < size; i++)
            index[i] = i;
        std::sort(index.begin(), index.end(), [&](Index i, Index j) { return m_target(data[i]) < m_target(data[j]); });

        // For SortRule::BothEnds, the eigenvalues are sorted according to the
        // SortRule::LargestAlge rule, so we need to move those smallest values to the left
        // The order would be
        //     Largest => Smallest => 2nd largest => 2nd smallest => ...
        // We keep this order since the first k values will always be
        // the wanted collection, no matter k is nev_updated (used in SymEigsBase::restart())
        // or is nev (used in SymEigsBase::sort_ritzpair())
        if (m_both_ends)
        {
            IndexArray index_copy(index);
            for (Index i = 0; i < size; i++)
            {
                // If i is even, pick values from the left (large values)
                // If i is odd, pick values from the right (small values)
                if (i % 2 == 0)
                    index[i] = index_copy[i / 2];
                else
                    index[i] = index_copy[size - 1 - i / 2];
            }
        }
        return index;
    }

    IndexArray argsort(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& values) const
    {
        return argsort(values.data(), values.size());
    }
};

}  // namespace Spectra

#endif  // SPECTRA_SELECTION_RULE_H
