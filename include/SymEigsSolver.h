// Copyright (C) 2015 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYM_EIGS_SOLVER_H
#define SYM_EIGS_SOLVER_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>     // std::vector
#include <cmath>      // std::abs, std::pow
#include <algorithm>  // std::min, std::copy
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::invalid_argument

#include "SelectionRule.h"
#include "CompInfo.h"
#include "LinAlg/UpperHessenbergQR.h"
#include "LinAlg/TridiagEigen.h"
#include "MatOp/DenseGenMatProd.h"
#include "MatOp/DenseSymShiftSolve.h"


namespace Spectra {


///
/// \defgroup EigenSolver Eigen Solvers
///
/// Eigen solvers for different types of problems.
///

///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for real symmetric matrices.
///
/// **Spectra** is designed to calculate a specified number (\f$k\f$)
/// of eigenvalues of a large square matrix (\f$A\f$). Usually \f$k\f$ is much
/// less than the size of the matrix (\f$n\f$), so that only a few eigenvalues
/// and eigenvectors are computed.
///
/// This class implements the eigen solver of a real symmetric matrix, but
/// rather than providing the whole matrix, the algorithm only requires the
/// matrix-vector multiplication operation of \f$A\f$. Therefore, users of
/// this solver need to supply a class that computes the result of \f$Av\f$
/// for any given vector \f$v\f$. The name of this class should be given to
/// the template parameter `OpType`, and instance of this class passed to
/// the constructor of SymEigsSolver.
///
/// If the matrix \f$A\f$ is already stored as a matrix object in **Eigen**,
/// for example `Eigen::MatrixXd`, then there is an easy way to construct such
/// matrix operation class, by using the built-in wrapper class DenseGenMatProd
/// which wraps an existing matrix object in **Eigen**. This is also the
/// default template parameter for SymEigsSolver.
///
/// If the users need to define their own matrix-vector multiplication operation
/// class, it should implement all the public member functions as in DenseGenMatProd.
///
/// \tparam Scalar        The element type of the matrix.
///                       Currently supported types are `float`, `double` and `long double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the requested eigenvalues, for example `LARGEST_MAGN`
///                       to retrieve eigenvalues with the largest magnitude.
///                       The full list of enumeration values can be found in
///                       \ref Enumerations.
/// \tparam OpType        The name of the matrix operation class. Users could either
///                       use the DenseGenMatProd wrapper class, or define their
///                       own that impelemnts all the public member functions as in
///                       DenseGenMatProd.
///
/// Below is an example that demonstrates the usage of this class.
///
/// \code{.cpp}
/// #include <Eigen/Core>
/// #include <SymEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
/// #include <iostream>
///
/// using namespace Spectra;
///
/// int main()
/// {
///     // We are going to calculate the eigenvalues of M
///     Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
///     Eigen::MatrixXd M = A + A.transpose();
///
///     // Construct matrix operation object using the wrapper class DenseGenMatProd
///     DenseGenMatProd<double> op(M);
///
///     // Construct eigen solver object, requesting the largest three eigenvalues
///     SymEigsSolver< double, LARGEST_ALGE, DenseGenMatProd<double> > eigs(&op, 3, 6);
///
///     // Initialize and compute
///     eigs.init();
///     int nconv = eigs.compute();
///
///     // Retrieve results
///     Eigen::VectorXd evalues;
///     if(eigs.info() == SUCCESSFUL)
///         evalues = eigs.eigenvalues();
///
///     std::cout << "Eigenvalues found:\n" << evalues << std::endl;
///
///     return 0;
/// }
/// \endcode
///
/// And here is an example for user-supplied matrix operation class.
///
/// \code{.cpp}
/// #include <Eigen/Core>
/// #include <SymEigsSolver.h>
/// #include <iostream>
///
/// using namespace Spectra;
///
/// // M = diag(1, 2, ..., 10)
/// class MyDiagonalTen
/// {
/// public:
///     int rows() { return 10; }
///     int cols() { return 10; }
///     // y_out = M * x_in
///     void perform_op(double *x_in, double *y_out)
///     {
///         for(int i = 0; i < rows(); i++)
///         {
///             y_out[i] = x_in[i] * (i + 1);
///         }
///     }
/// };
///
/// int main()
/// {
///     MyDiagonalTen op;
///     SymEigsSolver<double, LARGEST_ALGE, MyDiagonalTen> eigs(&op, 3, 6);
///     eigs.init();
///     eigs.compute();
///     if(eigs.info() == SUCCESSFUL)
///     {
///         Eigen::VectorXd evalues = eigs.eigenvalues();
///         // Will get (10, 9, 8)
///         std::cout << "Eigenvalues found:\n" << evalues << std::endl;
///     }
///
///     return 0;
/// }
/// \endcode
///
template < typename Scalar = double,
           int SelectionRule = LARGEST_MAGN,
           typename OpType = DenseGenMatProd<double> >
class SymEigsSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolArray;
    typedef Eigen::Map<Matrix> MapMat;
    typedef Eigen::Map<Vector> MapVec;
    typedef Eigen::SelfAdjointEigenSolver<Matrix> EigenSolver;

protected:
    OpType *m_op;          // object to conduct matrix operation,
                           // e.g. matrix-vector product

private:
    const int m_n;         // dimension of matrix A

protected:
    const int m_nev;       // number of eigenvalues requested

private:
    const int m_ncv;       // number of ritz values
    int m_nmatop;          // number of matrix operations called
    int m_niter;           // number of restarting iterations

    Matrix m_fac_V;        // V matrix in the Arnoldi factorization
    Matrix m_fac_H;        // H matrix in the Arnoldi factorization
    Vector m_fac_f;        // residual in the Arnoldi factorization

protected:
    Vector m_ritz_val;     // ritz values

private:
    Matrix m_ritz_vec;     // ritz vectors
    Vector m_ritz_est;     // last row of m_ritz_vec
    BoolArray m_ritz_conv; // indicator of the convergence of ritz values
    int m_info;            // status of the computation

    const Scalar m_prec;   // precision parameter used to test convergence
                           // m_prec = epsilon^(2/3)
                           // epsilon is the machine precision,
                           // e.g. ~= 1e-16 for the "double" type

    // Arnoldi factorization starting from step-k
    void factorize_from(int from_k, int to_m, const Vector &fk)
    {
        if(to_m <= from_k) return;

        m_fac_f = fk;

        Vector w(m_n);
        Scalar beta = m_fac_f.norm(), Hii = 0.0;
        // Keep the upperleft k x k submatrix of H and set other elements to 0
        m_fac_H.rightCols(m_ncv - from_k).setZero();
        m_fac_H.block(from_k, 0, m_ncv - from_k, from_k).setZero();
        for(int i = from_k; i <= to_m - 1; i++)
        {
            // v <- f / ||f||
            MapVec v(&m_fac_V(0, i), m_n); // The (i+1)-th column
            v.noalias() = m_fac_f / beta;
            m_fac_H(i, i - 1) = beta;

            // w <- A * v
            m_op->perform_op(v.data(), w.data());
            m_nmatop++;

            Hii = v.dot(w);
            m_fac_H(i - 1, i) = beta;
            m_fac_H(i, i) = Hii;

            //  f <- w - V * V' * w
            m_fac_f.noalias() = w - beta * m_fac_V.col(i - 1) - Hii * v;
            beta = m_fac_f.norm();

            // f/||f|| is going to be the next column of V, so we need to test
            // whether V' * (f/||f||) ~= 0
            MapMat V(m_fac_V.data(), m_n, i + 1); // The first (i+1) columns
            Vector Vf = V.transpose() * m_fac_f;
            if(Vf.cwiseAbs().maxCoeff() > m_prec * beta)
            {
                // f <- f - V * V' * f
                m_fac_f.noalias() -= V * Vf;
                beta = m_fac_f.norm();
            }
        }
    }

    // Implicitly restarted Arnoldi factorization
    void restart(int k)
    {
        if(k >= m_ncv)
            return;

        TridiagQR<Scalar> decomp;
        Matrix Q = Matrix::Identity(m_ncv, m_ncv);

        for(int i = k; i < m_ncv; i++)
        {
            // QR decomposition of H-mu*I, mu is the shift
            m_fac_H.diagonal().array() -= m_ritz_val[i];
            decomp.compute(m_fac_H);

            // Q -> Q * Qi
            decomp.apply_YQ(Q);
            // H -> Q'HQ
            // Since QR = H - mu * I, we have H = QR + mu * I
            // and therefore Q'HQ = RQ + mu * I
            m_fac_H = decomp.matrix_RQ();
            m_fac_H.diagonal().array() += m_ritz_val[i];
        }
        // V -> VQ, only need to update the first k+1 columns
        // Q has some elements being zero
        // The first (ncv - k + i) elements of the i-th column of Q are non-zero
        Matrix Vs(m_n, k + 1);
        int nnz;
        for(int i = 0; i < k; i++)
        {
            nnz = m_ncv - k + i + 1;
            MapMat V(m_fac_V.data(), m_n, nnz);
            MapVec q(&Q(0, i), nnz);
            Vs.col(i).noalias() = V * q;
        }
        Vs.col(k).noalias() = m_fac_V * Q.col(k);
        m_fac_V.leftCols(k + 1).noalias() = Vs;

        Vector fk = m_fac_f * Q(m_ncv - 1, k - 1) + m_fac_V.col(k) * m_fac_H(k, k - 1);
        factorize_from(k, m_ncv, fk);
        retrieve_ritzpair();
    }

    // Calculate the number of converged Ritz values
    int num_converged(Scalar tol)
    {
        // thresh = tol * max(m_prec, abs(theta)), theta for ritz value
        Array thresh = tol * m_ritz_val.head(m_nev).array().abs().max(m_prec);
        Array resid =  m_ritz_est.head(m_nev).array().abs() * m_fac_f.norm();
        // Converged "wanted" ritz values
        m_ritz_conv = (resid < thresh);

        return m_ritz_conv.cast<int>().sum();
    }

    // Return the adjusted nev for restarting
    int nev_adjusted(int nconv)
    {
        int nev_new = m_nev;

        for(int i = m_nev; i < m_ncv; i++)
            if(std::abs(m_ritz_est[i]) < m_prec)  nev_new++;

        // Adjust nev_new, according to dsaup2.f line 677~684 in ARPACK
        nev_new += std::min(nconv, (m_ncv - nev_new) / 2);
        if(nev_new == 1 && m_ncv >= 6)
            nev_new = m_ncv / 2;
        else if(nev_new == 1 && m_ncv > 2)
            nev_new = 2;

        return nev_new;
    }

    // Retrieve and sort ritz values and ritz vectors
    void retrieve_ritzpair()
    {
        TridiagEigen<Scalar> decomp(m_fac_H);
        Vector evals = decomp.eigenvalues();
        Matrix evecs = decomp.eigenvectors();

        SortEigenvalue<Scalar, SelectionRule> sorting(evals.data(), evals.size());
        std::vector<int> ind = sorting.index();

        // For BOTH_ENDS, the eigenvalues are sorted according
        // to the LARGEST_ALGE rule, so we need to move those smallest
        // values to the left
        // The order would be
        // Largest => Smallest => 2nd largest => 2nd smallest => ...
        // We keep this order since the first k values will always be
        // the wanted collection, no matter k is nev_updated (used in restart())
        // or is nev (used in sort_ritzpair())
        if(SelectionRule == BOTH_ENDS)
        {
            std::vector<int> ind_copy(ind);
            for(int i = 0; i < m_ncv; i++)
            {
                // If i is even, pick values from the left (large values)
                // If i is odd, pick values from the right (small values)
                if(i % 2 == 0)
                    ind[i] = ind_copy[i / 2];
                else
                    ind[i] = ind_copy[m_ncv - 1 - i / 2];
            }
        }

        // Copy the ritz values and vectors to m_ritz_val and m_ritz_vec, respectively
        for(int i = 0; i < m_ncv; i++)
        {
            m_ritz_val[i] = evals[ind[i]];
            m_ritz_est[i] = evecs(m_ncv - 1, ind[i]);
        }
        for(int i = 0; i < m_nev; i++)
        {
            m_ritz_vec.col(i) = evecs.col(ind[i]);
        }
    }

protected:
    // Sort the first nev Ritz pairs in the specified order
    // This is used to return the final results
    virtual void sort_ritzpair(int sort_rule)
    {
        // First make sure that we have a valid index vector
        SortEigenvalue<Scalar, LARGEST_ALGE> sorting(m_ritz_val.data(), m_nev);
        std::vector<int> ind = sorting.index();

        switch(sort_rule)
        {
            case LARGEST_ALGE:
                break;
            case LARGEST_MAGN:
            {
                SortEigenvalue<Scalar, LARGEST_MAGN> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            case SMALLEST_ALGE:
            {
                SortEigenvalue<Scalar, SMALLEST_ALGE> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            case SMALLEST_MAGN:
            {
                SortEigenvalue<Scalar, SMALLEST_MAGN> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            default:
                throw std::invalid_argument("unsupported sorting rule");
        }

        Vector new_ritz_val(m_ncv);
        Matrix new_ritz_vec(m_ncv, m_nev);
        BoolArray new_ritz_conv(m_nev);

        for(int i = 0; i < m_nev; i++)
        {
            new_ritz_val[i] = m_ritz_val[ind[i]];
            new_ritz_vec.col(i) = m_ritz_vec.col(ind[i]);
            new_ritz_conv[i] = m_ritz_conv[ind[i]];
        }

        m_ritz_val.swap(new_ritz_val);
        m_ritz_vec.swap(new_ritz_vec);
        m_ritz_conv.swap(new_ritz_conv);
    }

public:
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op_  Pointer to the matrix operation object, which should implement
    ///             the matrix-vector multiplication operation of \f$A\f$:
    ///             calculating \f$Ay\f$ for any vector \f$y\f$. Users could either
    ///             create the object from the DenseGenMatProd wrapper class, or
    ///             define their own that impelemnts all the public member functions
    ///             as in DenseGenMatProd.
    /// \param nev_ Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv_ Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv_` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\cdot nev\f$.
    ///
    SymEigsSolver(OpType *op_, int nev_, int ncv_) :
        m_op(op_),
        m_n(m_op->rows()),
        m_nev(nev_),
        m_ncv(ncv_ > m_n ? m_n : ncv_),
        m_nmatop(0),
        m_niter(0),
        m_info(NOT_COMPUTED),
        m_prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0) / 3))
    {
        if(nev_ < 1 || nev_ > m_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

        if(ncv_ <= nev_ || ncv_ > m_n)
            throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");
    }

    ///
    /// Providing the initial residual vector for the algorithm.
    ///
    /// \param init_resid Pointer to the initial residual vector.
    ///
    /// **Spectra** (and also **ARPACK**) uses an iterative algorithm
    /// to find eigenvalues. This function allows the user to provide the initial
    /// residual vector.
    ///
    void init(const Scalar *init_resid)
    {
        // Reset all matrices/vectors to zero
        m_fac_V.resize(m_n, m_ncv);
        m_fac_H.resize(m_ncv, m_ncv);
        m_fac_f.resize(m_n);
        m_ritz_val.resize(m_ncv);
        m_ritz_vec.resize(m_ncv, m_nev);
        m_ritz_est.resize(m_ncv);
        m_ritz_conv.resize(m_nev);

        m_fac_V.setZero();
        m_fac_H.setZero();
        m_fac_f.setZero();
        m_ritz_val.setZero();
        m_ritz_vec.setZero();
        m_ritz_est.setZero();
        m_ritz_conv.setZero();

        m_nmatop = 0;
        m_niter = 0;

        Vector v(m_n);
        std::copy(init_resid, init_resid + m_n, v.data());
        Scalar vnorm = v.norm();
        if(vnorm < m_prec)
            throw std::invalid_argument("initial residual vector cannot be zero");
        v /= vnorm;

        Vector w(m_n);
        m_op->perform_op(v.data(), w.data());
        m_nmatop++;

        m_fac_H(0, 0) = v.dot(w);
        m_fac_f = w - v * m_fac_H(0, 0);
        m_fac_V.col(0) = v;
    }

    ///
    /// Providing a random initial residual vector.
    ///
    /// This overloaded function generates a random initial residual vector
    /// for the algorithm. Elements in the vector follow independent Uniform(-0.5, 0.5)
    /// distributions.
    ///
    void init()
    {
        Vector init_resid = Vector::Random(m_n);
        init_resid.array() -= 0.5;
        init(init_resid.data());
    }

    ///
    /// Conducting the major computation procedure.
    ///
    /// \param maxit      Maximum number of iterations allowed in the algorithm.
    /// \param tol        Precision parameter for the calculated eigenvalues.
    /// \param sort_rule  Rule to sort the eigenvalues and eigenvectors.
    ///                   Supported values are
    ///                   `Spectra::LARGEST_ALGE`, `Spectra::LARGEST_MAGN`,
    ///                   `Spectra::SMALLEST_ALGE` and `Spectra::SMALLEST_MAGN`,
    ///                   for example `LARGEST_ALGE` indicates that largest eigenvalues
    ///                   come first. Note that this argument is only used to
    ///                   **sort** the final result, and the **selection** rule
    ///                   (e.g. selecting the largest or smallest eigenvalues in the
    ///                   full spectrum) is specified by the template parameter
    ///                   `SelectionRule` of SymEigsSolver.
    ///
    /// \return Number of converged eigenvalues.
    ///
    int compute(int maxit = 1000, Scalar tol = 1e-10, int sort_rule = LARGEST_ALGE)
    {
        // The m-step Arnoldi factorization
        factorize_from(1, m_ncv, m_fac_f);
        retrieve_ritzpair();
        // Restarting
        int i, nconv = 0, nev_adj;
        for(i = 0; i < maxit; i++)
        {
            nconv = num_converged(tol);
            if(nconv >= m_nev)
                break;

            nev_adj = nev_adjusted(nconv);
            restart(nev_adj);
        }
        // Sorting results
        sort_ritzpair(sort_rule);

        m_niter += i + 1;
        m_info = (nconv >= m_nev) ? SUCCESSFUL : NOT_CONVERGING;

        return std::min(m_nev, nconv);
    }

    ///
    /// Retuning the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
    int info() { return m_info; }

    ///
    /// Returning the number of iterations used in the computation.
    ///
    int num_iterations() { return m_niter; }

    ///
    /// Returning the number of matrix operations used in the computation.
    ///
    int num_operations() { return m_nmatop; }

    ///
    /// Returning the converged eigenvalues.
    ///
    /// \return A vector containing the eigenvalues.
    /// Returned vector type will be `Eigen::Vector<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Vector eigenvalues()
    {
        int nconv = m_ritz_conv.cast<int>().sum();
        Vector res(nconv);

        if(!nconv)
            return res;

        int j = 0;
        for(int i = 0; i < m_nev; i++)
        {
            if(m_ritz_conv[i])
            {
                res[j] = m_ritz_val[i];
                j++;
            }
        }

        return res;
    }

    ///
    /// Returning the eigenvectors associated with the converged eigenvalues.
    ///
    /// \param nvec The number of eigenvectors to return.
    ///
    /// \return A matrix containing the eigenvectors.
    /// Returned matrix type will be `Eigen::Matrix<Scalar, ...>`,
    /// depending on the template parameter `Scalar` defined.
    ///
    Matrix eigenvectors(int nvec)
    {
        int nconv = m_ritz_conv.cast<int>().sum();
        nvec = std::min(nvec, nconv);
        Matrix res(m_n, nvec);

        if(!nvec)
            return res;

        Matrix ritz_vec_conv(m_ncv, nvec);
        int j = 0;
        for(int i = 0; i < m_nev && j < nvec; i++)
        {
            if(m_ritz_conv[i])
            {
                ritz_vec_conv.col(j) = m_ritz_vec.col(i);
                j++;
            }
        }

        res.noalias() = m_fac_V * ritz_vec_conv;

        return res;
    }

    ///
    /// Returning all converged eigenvectors.
    ///
    Matrix eigenvectors()
    {
        return eigenvectors(m_nev);
    }
};





///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for real symmetric matrices using
/// the **shift-and-invert mode**. The background information of the symmetric
/// eigen solver is documented in the SymEigsSolver class. Here we focus on
/// explaining the shift-and-invert mode.
///
/// The shift-and-invert mode is based on the following fact:
/// If \f$\lambda\f$ and \f$x\f$ are a pair of eigenvalue and eigenvector of
/// matrix \f$A\f$, such that \f$Ax=\lambda x\f$, then for any \f$\sigma\f$,
/// we have
/// \f[(A-\sigma I)^{-1}x=\nu x\f]
/// where
/// \f[\nu=\frac{1}{\lambda-\sigma}\f]
/// which indicates that \f$(\nu, x)\f$ is an eigenpair of the matrix
/// \f$(A-\sigma I)^{-1}\f$.
///
/// Therefore, if we pass the matrix operation \f$(A-\sigma I)^{-1}y\f$
/// (rather than \f$Ay\f$) to the eigen solver, then we would get the desired
/// values of \f$\nu\f$, and \f$\lambda\f$ can also be easily obtained by noting
/// that \f$\lambda=\sigma+\nu^{-1}\f$.
///
/// The reason why we need this type of manipulation is that
/// the algorithm of **Spectra** (and also **ARPACK**)
/// is good at finding eigenvalues with large magnitude, but may fail in looking
/// for eigenvalues that are close to zero. However, if we really need them, we
/// can set \f$\sigma=0\f$, find the largest eigenvalues of \f$A^{-1}\f$, and then
/// transform back to \f$\lambda\f$, since in this case largest values of \f$\nu\f$
/// implies smallest values of \f$\lambda\f$.
///
/// To summarize, in the shift-and-invert mode, the selection rule will apply to
/// \f$\nu=1/(\lambda-\sigma)\f$ rather than \f$\lambda\f$. So a selection rule
/// of `LARGEST_MAGN` combined with shift \f$\sigma\f$ will find eigenvalues of
/// \f$A\f$ that are closest to \f$\sigma\f$. But note that the eigenvalues()
/// method will always return the eigenvalues in the original problem (i.e.,
/// returning \f$\lambda\f$ rather than \f$\nu\f$), and eigenvectors are the
/// same for both the original problem and the shifted-and-inverted problem.
///
/// \tparam Scalar        The element type of the matrix.
///                       Currently supported types are `float`, `double` and `long double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the shifted-and-inverted eigenvalues.
///                       The full list of enumeration values can be found in
///                       \ref Enumerations.
/// \tparam OpType        The name of the matrix operation class. Users could either
///                       use the DenseSymShiftSolve wrapper class, or define their
///                       own that impelemnts all the public member functions as in
///                       DenseSymShiftSolve.
///
/// Below is an example that illustrates the use of the shift-and-invert mode:
///
/// \code{.cpp}
/// #include <Eigen/Core>
/// #include <SymEigsSolver.h>  // Also includes <MatOp/DenseSymShiftSolve.h>
/// #include <iostream>
///
/// using namespace Spectra;
///
/// int main()
/// {
///     // A size-10 diagonal matrix with elements 1, 2, ..., 10
///     Eigen::MatrixXd M = Eigen::MatrixXd::Zero(10, 10);
///     for(int i = 0; i < M.rows(); i++)
///         M(i, i) = i + 1;
///
///     // Construct matrix operation object using the wrapper class
///     DenseSymShiftSolve<double> op(M);
///
///     // Construct eigen solver object with shift 0
///     // This will find eigenvalues that are closest to 0
///     SymEigsShiftSolver< double, LARGEST_MAGN,
///                         DenseSymShiftSolve<double> > eigs(&op, 3, 6, 0.0);
///
///     eigs.init();
///     eigs.compute();
///     if(eigs.info() == SUCCESSFUL)
///     {
///         Eigen::VectorXd evalues = eigs.eigenvalues();
///         // Will get (3.0, 2.0, 1.0)
///         std::cout << "Eigenvalues found:\n" << evalues << std::endl;
///     }
///
///     return 0;
/// }
/// \endcode
///
/// Also an example for user-supplied matrix shift-solve operation class:
///
/// \code{.cpp}
/// #include <Eigen/Core>
/// #include <SymEigsSolver.h>
/// #include <iostream>
///
/// using namespace Spectra;
///
/// // M = diag(1, 2, ..., 10)
/// class MyDiagonalTenShiftSolve
/// {
/// private:
///     double sigma_;
/// public:
///     int rows() { return 10; }
///     int cols() { return 10; }
///     void set_shift(double sigma) { sigma_ = sigma; }
///     // y_out = inv(A - sigma * I) * x_in
///     // inv(A - sigma * I) = diag(1/(1-sigma), 1/(2-sigma), ...)
///     void perform_op(double *x_in, double *y_out)
///     {
///         for(int i = 0; i < rows(); i++)
///         {
///             y_out[i] = x_in[i] / (i + 1 - sigma_);
///         }
///     }
/// };
///
/// int main()
/// {
///     MyDiagonalTenShiftSolve op;
///     // Find three eigenvalues that are closest to 3.14
///     SymEigsShiftSolver<double, LARGEST_MAGN,
///                        MyDiagonalTenShiftSolve> eigs(&op, 3, 6, 3.14);
///     eigs.init();
///     eigs.compute();
///     if(eigs.info() == SUCCESSFUL)
///     {
///         Eigen::VectorXd evalues = eigs.eigenvalues();
///         // Will get (4.0, 3.0, 2.0)
///         std::cout << "Eigenvalues found:\n" << evalues << std::endl;
///     }
///
///     return 0;
/// }
/// \endcode
///
template <typename Scalar = double,
          int SelectionRule = LARGEST_MAGN,
          typename OpType = DenseSymShiftSolve<double> >
class SymEigsShiftSolver: public SymEigsSolver<Scalar, SelectionRule, OpType>
{
private:
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

    Scalar sigma;

    // First transform back the ritz values, and then sort
    void sort_ritzpair(int sort_rule)
    {
        Array m_ritz_val_org = Scalar(1.0) / this->m_ritz_val.head(this->m_nev).array() + sigma;
        this->m_ritz_val.head(this->m_nev) = m_ritz_val_org;
        SymEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair(sort_rule);
    }
public:
    ///
    /// Constructor to create a eigen solver object using the shift-and-invert mode.
    ///
    /// \param op_    Pointer to the matrix operation object, which should implement
    ///               the shift-solve operation of \f$A\f$: calculating
    ///               \f$(A-\sigma I)^{-1}y\f$ for any vector \f$y\f$. Users could either
    ///               create the object from the DenseSymShiftSolve wrapper class, or
    ///               define their own that impelemnts all the public member functions
    ///               as in DenseSymShiftSolve.
    /// \param nev_   Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///               where \f$n\f$ is the size of matrix.
    /// \param ncv_   Parameter that controls the convergence speed of the algorithm.
    ///               Typically a larger `ncv_` means faster convergence, but it may
    ///               also result in greater memory use and more matrix operations
    ///               in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///               and is advised to take \f$ncv \ge 2\cdot nev\f$.
    /// \param sigma_ The value of the shift.
    ///
    SymEigsShiftSolver(OpType *op_, int nev_, int ncv_, Scalar sigma_) :
        SymEigsSolver<Scalar, SelectionRule, OpType>(op_, nev_, ncv_),
        sigma(sigma_)
    {
        this->m_op->set_shift(sigma);
    }
};


} // namespace Spectra

#endif // SYM_EIGS_SOLVER_H
