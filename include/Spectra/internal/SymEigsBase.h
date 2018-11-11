// Copyright (C) 2018 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SYM_EIGS_BASE_H
#define SYM_EIGS_BASE_H

#include "../MatOp/internal/ArnoldiOp.h"
#include "../LinAlg/Lanczos.h"

namespace Spectra {


template < typename Scalar,
           typename OpType,
           typename BOpType >
class SymEigsBase
{
private:
    typedef ArnoldiOp<Scalar, OpType, BOpType> ArnoldiOpType;
    typedef Lanczos<Scalar, ArnoldiOpType> LanczosFac;

protected:
    LanczosFac m_fac;  // Lanczos factorization

public:
    SymEigsBase(OpType* op, BOpType* Bop, int ncv) :
        m_fac(ArnoldiOpType(op, Bop), ncv)
    {}
};


} // namespace Spectra

#endif // SYM_EIGS_BASE_H
