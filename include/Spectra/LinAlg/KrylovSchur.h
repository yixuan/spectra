// Author: David Kriebel <dotnotlock@gmail.com>

#ifndef SPECTRA_KRYLOVSCHUR_H
#define SPECTRA_KRYLOVSCHUR_H

#include <Eigen/Core>
#include <cmath>      // std::sqrt
#include <utility>    // std::move
#include <stdexcept>  // std::invalid_argument

#include "../MatOp/internal/ArnoldiOp.h"
#include "../Util/TypeTraits.h"
#include "../Util/SimpleRandom.h"
#include "../Util/CompInfo.h"

#include <iostream>

namespace Spectra {

// (Krylov-Schur specific) modified Arnoldi decompostion A * V = V * H + f * e'
// A: n x n
// V: n x k
// H: k x k
// f: n x 1
// e: [0, ..., 0, 1]
// V and H are allocated of dimension m, so the maximum value of k is m
template <typename Scalar, typename ArnoldiOpType>
class KrylovSchur : public Arnoldi<Scalar, ArnoldiOpType>
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapVec = Eigen::Map<Vector>;
    using MapConstMat = Eigen::Map<const Matrix>;
    using MapConstVec = Eigen::Map<const Vector>;

    bool m_stop_algorithm = false;  // convergence indicator flag

    using Arnoldi<Scalar, ArnoldiOpType>::m_op;
    using Arnoldi<Scalar, ArnoldiOpType>::m_n;
    using Arnoldi<Scalar, ArnoldiOpType>::m_m;
    using Arnoldi<Scalar, ArnoldiOpType>::m_k;
    using Arnoldi<Scalar, ArnoldiOpType>::m_beta;
    using Arnoldi<Scalar, ArnoldiOpType>::m_near_0;
    using Arnoldi<Scalar, ArnoldiOpType>::m_eps;

protected:
    using Arnoldi<Scalar, ArnoldiOpType>::m_fac_V;
    using Arnoldi<Scalar, ArnoldiOpType>::m_fac_H;
    using Arnoldi<Scalar, ArnoldiOpType>::m_fac_f;

    Vector m_fac_v;   // current v vector (residual)
    CompInfo m_info;  // status of the computation

    virtual bool robust_reorthogonalize(MapConstMat& Vjj, Vector& f, Scalar& fnorm, const Index jj, const Index seed, Vector& wIn)
    {
        // reorthogonalization to insure that vector f is orthogonal to the coloumn space Vjj (Reference: G. W. Stewart; A Krylov-Schur Algorithm for Large Eigenproblems; 2000)
        // Classical Gram-Schmidt orthogonalization with reorthogonalization (Reference: G. W. Stewart; Matrix algorithms. Volume 1, Basic decompositions; 1998; page 287, Algorithm 4.1.13)

        Scalar normf0 = f.norm();  // normr0 = sqrt(abs(r'*(applyM(r))));
        Vector w = wIn;            // copy wIn to w

        bool stopAlgorithm = false;  // return value which tops the inner iterations

        // Reorthogonalize :
        Vector dw = Vjj.transpose() * f;
        f.noalias() -= Vjj * dw;
        w.noalias() += dw;
        fnorm = f.norm();

        // Classical Gram-Schmidt Algorithm
        int numReorths = 1;
        while (fnorm <= (1 / sqrt(2)) * normf0 && numReorths < 5)
        {
            dw.noalias() = Vjj.transpose() * f;
            f.noalias() -= Vjj * dw;
            w.noalias() += dw;
            normf0 = fnorm;
            fnorm = f.norm();
            numReorths++;
        }

        // check for invariant subspace as a protection against loss of orthogonality
        if (fnorm <= (1 / sqrt(2)) * normf0)
        {
            // Cannot Reorthogonalize, invariant subspace found.
            fnorm = 0;
            w.noalias() = wIn;

            // Try a random restart
            stopAlgorithm = true;

            for (int restart = 1; restart <= 3; restart++)
            {
                // Do a random restart: Will try at most three times before stoppping the algorithm
                SimpleRandom<Scalar> rng(seed + 123 * restart);
                rng.random_vec(f);  // get a new random f vector

                // Orthogonalize f
                f -= Vjj * (Vjj.transpose() * f);
                f.normalize();

                // Reorthogonalize if necessary
                stopAlgorithm = true;
                for (int reorth = 1; reorth <= 5; reorth++)
                {
                    // Check orthogonality
                    Vector Vf = Vjj.transpose() * f;

                    if (abs(f.norm() - 1) <= 1e-10 && Vf.cwiseAbs().maxCoeff() <= 1e-10)
                    {
                        stopAlgorithm = false;
                        break;
                    }

                    // Reorthogonalize
                    f.noalias() -= Vjj * Vf;
                    f.normalize();
                }

                if (!stopAlgorithm)
                    break;
            }
        }
        else
        {
            f.normalize();
        }

        wIn.noalias() = w;  // assign the new w vector as output
        return stopAlgorithm;
    }

public:
    // Copy an ArnoldiOp
    template <typename T>
    KrylovSchur(T& op, Index m) :
        Arnoldi<Scalar, ArnoldiOpType>(op, m)
    {}

    // Move an ArnoldiOp
    template <typename T>
    KrylovSchur(T&& op, Index m) :
        Arnoldi<Scalar, ArnoldiOpType>(std::forward<T>(op), m)
    {}

    // reference to internal structures
    Matrix& matrix_V() { return m_fac_V; }
    Matrix& matrix_H() { return m_fac_H; }
    Vector& vector_f() { return m_fac_f; }

    // Initialize with an operator and an initial vector
    void init(MapConstVec& v0, Index& op_counter)
    {
        m_fac_V.resize(m_n, m_m);
        m_fac_H.resize(m_m + 1, m_m);
        m_fac_f.resize(m_n);
        m_fac_v.resize(m_n);

        m_fac_H.setZero();
        m_fac_V.setZero();

        // Verify the initial vector
        const Scalar v0norm = v0.norm();
        if (v0norm < m_near_0)
            throw std::invalid_argument("initial residual vector cannot be zero");

        // Normalize
        m_fac_v.noalias() = v0.normalized();
        m_k = 0;
    }

    // Krylov factorization starting from step-k
    void factorize_from(Index from_k, Index to_m, Index& op_counter)
    {
        using std::sqrt;

        m_stop_algorithm = true;

        if (to_m <= from_k)
            return;

        if (from_k > m_k)
        {
            std::string msg = "KrylovSchur: from_k (= " + std::to_string(from_k) +
                ") is larger than the current subspace dimension (= " + std::to_string(m_k) + ")";
            throw std::invalid_argument(msg);
        }

        const Scalar beta_thresh = m_eps * sqrt(Scalar(m_n));

        // Keep the upperleft k+1 x k submatrix of H and set other elements to 0
        m_fac_H.rightCols(m_m - from_k).setZero();
        m_fac_H.block(from_k + 1, 0, m_m - from_k - 1, from_k).setZero();

        for (Index i = from_k; i <= to_m - 1; i++)
        {
            m_fac_V.col(i).noalias() = m_fac_v;               // V(:, jj) = v;
            m_op.perform_op(m_fac_v.data(), m_fac_f.data());  // r = applyOP(applyM(v));
            op_counter++;

            MapConstMat Vjj(m_fac_V.data(), m_n, i + 1);  // Vjj = matlab.internal.math.viewColumns(V, jj);

            Vector w(i + 1);
            w.noalias() = Vjj.transpose() * m_fac_f;  // w = Vjj' * applyM(r);
            m_fac_f.noalias() -= Vjj * w;             // r = r - Vjj * w;

            // Reorthogonalize
            m_stop_algorithm = robust_reorthogonalize(Vjj, m_fac_f, m_beta, i, 2 * i, w);

            if (m_stop_algorithm)
            {
                return;
            }

            // Save data
            m_fac_v.noalias() = m_fac_f;
            m_fac_H.block(0, i, i + 1, 1).noalias() = w;  // H = [H w; zeros(1, jj-1) normRes];
            m_fac_H(i + 1, i) = m_beta;
        }

        // Indicate that this is a step-m factorization
        m_k = to_m;
    }

    // Updating
    void swap_H(Matrix& other)
    {
        m_fac_H.swap(other);
    }

    void swap_V(Matrix& other)
    {
        m_fac_V.swap(other);
    }

    void swap_f(Vector& other)
    {
        m_fac_f.swap(other);
    }

    ///
    /// Returns the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
    CompInfo info()
    {
        return m_stop_algorithm ? CompInfo::NotConverging : CompInfo::Successful;
    }
};

}  // namespace Spectra

#endif  // SPECTRA_KRYLOVSCHUR_H
