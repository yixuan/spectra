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

#include <iostream>

namespace Spectra {

// Krylov decompostion A * V = V * H + f * e'
// A: n x n
// V: n x k
// H: k x k
// f: n x 1
// e: [0, ..., 0, 1]
// V and H are allocated of dimension m, so the maximum value of k is m
template <typename Scalar, typename ArnoldiOpType>
class KrylovSchur
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapVec = Eigen::Map<Vector>;
    using MapConstMat = Eigen::Map<const Matrix>;
    using MapConstVec = Eigen::Map<const Vector>;

protected:
    // A very small value, but 1.0 / m_near_0 does not overflow
    // ~= 1e-307 for the "double" type
    static constexpr Scalar m_near_0 = TypeTraits<Scalar>::min() * Scalar(10);
    // The machine precision, ~= 1e-16 for the "double" type
    static constexpr Scalar m_eps = TypeTraits<Scalar>::epsilon();

    ArnoldiOpType m_op;  // Operators for the Arnoldi factorization
    const Index m_n;     // dimension of A
    const Index m_m;     // maximum dimension of subspace V
    Index m_k;           // current dimension of subspace V
    Matrix m_fac_V;      // V matrix in the Arnoldi factorization
    Matrix m_fac_H;      // H matrix in the Arnoldi factorization
    Vector m_fac_f;      // residual in the Arnoldi factorization
    Vector m_fac_v;      // current v vector (residual)
    Scalar m_beta;       // ||f||, B-norm of f

    bool robust_reorthogonalize(MapConstMat& Vjj, Vector& f, Scalar& fnorm, const Index jj, const Index seed, Vector& wIn)
    {
        Scalar normf0 = f.norm(); // normr0 = sqrt(abs(r'*(applyM(r))));
        Vector w(jj + 1);         // copy wIn to w
        w.noalias() = wIn;

        bool stopAlgorithm = false;
        // Reorthogonalize :
        Vector dw(jj + 1);
        dw.noalias() = Vjj.transpose() * f;
        f.noalias() -= Vjj * dw;
        w.noalias() += dw;
        fnorm = f.norm();

        int numReorths = 1;
        while (fnorm <= (1 / sqrt(2)) * normf0 && numReorths < 5)
        {
            dw.noalias() = Vjj.transpose() * f;
            f.noalias() -= Vjj * dw;
            w.noalias() += dw;
            normf0 = fnorm;
            fnorm = f.norm();
            numReorths = numReorths + 1;
        }
        
        if (fnorm <= (1 / sqrt(2)) * normf0)
        {
            // Cannot Reorthogonalize, invariant subspace found.
            fnorm = 0;
            w.noalias() = wIn;
    
            // Try a random restart
            stopAlgorithm = true;
    
            for(int restart = 1; restart<=3; restart++)
            {
                // Do a random restart: Will try at most three times
                SimpleRandom<Scalar> rng(seed + 123 * restart);
                rng.random_vec(f);
        
                // Orthogonalize r
                f.noalias() -= Vjj * (Vjj.transpose() * f);
                Scalar fMf = f.norm();
                f.normalize();
        
                // Re-orthogonalize if necessary
                stopAlgorithm = true;
                for (int reorth = 1; reorth <= 5; reorth++)
                {
                    // Check orthogonality
                    Vector VMf(jj + 1);
                    VMf.noalias() = Vjj.transpose() * f;
                    fMf = f.norm();
            
                    Scalar ortho_err = VMf.cwiseAbs().maxCoeff();
                    if (abs(fMf - 1) <= 1e-10 && ortho_err <= 1e-10)
                    {
                        stopAlgorithm = false;
                        break;
                    }
            
                    // Re-orthogonalize
                    f.noalias() -= Vjj * VMf;
                    f.normalize();
                }
        
                if(!stopAlgorithm)
                    break;
            }
        }
        else
        {
            f.normalize();
        }

        wIn.noalias() = w;
        return stopAlgorithm;
    }

public:
    // Copy an ArnoldiOp
    KrylovSchur(const ArnoldiOpType& op, Index m) :
        m_op(op), m_n(op.rows()), m_m(m), m_k(0)
    {}

    // Move an ArnoldiOp
    KrylovSchur(ArnoldiOpType&& op, Index m) :
        m_op(std::move(op)), m_n(op.rows()), m_m(m), m_k(0)
    {}

    // reference to internal structures
    Matrix& matrix_V() { return m_fac_V; }
    Matrix& matrix_H() { return m_fac_H; }
    Vector& vector_f() { return m_fac_f; }

    // Const-reference to internal structures
    const Matrix& matrix_V() const { return m_fac_V; }
    const Matrix& matrix_H() const { return m_fac_H; }
    const Vector& vector_f() const { return m_fac_f; }
    Scalar f_norm() const { return m_beta; }
    Index subspace_dim() const { return m_k; }

    // Initialize with an operator and an initial vector
    void init(MapConstVec& v0, Index& op_counter)
    {
        m_fac_V.resize(m_n, m_m);
        m_fac_H.resize(m_m+1, m_m);
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
    virtual bool factorize_from(Index from_k, Index to_m, Index& op_counter)
    {
        using std::sqrt;

        if (to_m <= from_k)
            return true;

        if (from_k > m_k)
        {
            std::string msg = "KrylovSchur: from_k (= " + std::to_string(from_k) +
                ") is larger than the current subspace dimension (= " + std::to_string(m_k) + ")";
            throw std::invalid_argument(msg);
        }

        const Scalar beta_thresh = m_eps * sqrt(Scalar(m_n));

        // Keep the upperleft k+1 x k submatrix of H and set other elements to 0
        m_fac_H.rightCols(m_m - from_k).setZero();
        m_fac_H.block(from_k+1, 0, m_m - from_k - 1, from_k).setZero();

        bool stopAlgorithm = false;
        for (Index i = from_k; i <= to_m - 1; i++)
        {
            m_fac_V.col(i).noalias() = m_fac_v;                 // V(:, jj) = v;
            m_op.perform_op(m_fac_v.data(), m_fac_f.data());  // r = applyOP(applyM(v));
            op_counter++;

            MapConstMat Vjj(m_fac_V.data(), m_n, i+1); // Vjj = matlab.internal.math.viewColumns(V, jj);

            Vector w(i + 1);
            w.noalias() = Vjj.transpose() * m_fac_f;   // w = Vjj' * applyM(r);
            m_fac_f.noalias() -= Vjj * w;              // r = r - Vjj * w;

            // Reorthogonalize
            stopAlgorithm = robust_reorthogonalize(Vjj, m_fac_f, m_beta, i, 2 * i, w);

            if (stopAlgorithm)
            {
                //    U = [];
                //    d = [];
                //    isNotConverged = false(0, 1);
                return stopAlgorithm;
            }

            // Save data
            m_fac_v.noalias() = m_fac_f;
            m_fac_H.block(0, i, i + 1, 1).noalias() = w;  // H = [H w; zeros(1, jj-1) normRes];
            m_fac_H(i + 1, i) = m_beta;
        }

        // Indicate that this is a step-m factorization
        m_k = to_m;

        return stopAlgorithm;
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
};

}  // namespace Spectra

#endif  // SPECTRA_KRYLOVSCHUR_H
