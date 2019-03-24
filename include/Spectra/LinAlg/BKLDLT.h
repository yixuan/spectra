// Copyright (C) 2019 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BK_LDLT_H
#define BK_LDLT_H

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace Spectra {


template <typename Scalar = double>
class BKLDLT
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<Vector> MapVec;
    typedef Eigen::Map<const Vector> MapConstVec;

    typedef typename Matrix::Index Index;

    typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> IntVector;
    typedef Eigen::Ref<Matrix> GenericMatrix;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;

    Index m_n;
    Vector m_data;
    std::vector<Scalar*> m_mat;  // m_mat represents a lower-triangular matrix
                                 // m_mat[i] points to the head of the i-th column
    IntVector m_perm;            // [-2, -1, 3, 1, 4, 5]: 0 <-> 2, 1 <-> 1, 2 <-> 3, 3 <-> 1, 4 <-> 4, 5 <-> 5
    std::vector< std::pair<Index, Index> > m_permc;  // compressed version of m_perm: [(0, 2), (2, 3), (3, 1)]

    bool m_computed;

    void compute_pointer()
    {
        m_mat.clear();
        m_mat.reserve(m_n);
        Scalar* head = m_data.data();

        for(Index i = 0; i < m_n; i++)
        {
            m_mat.push_back(head);
            head += (m_n - i);
        }
    }

    void copy_data(ConstGenericMatrix& mat, int uplo = Eigen::Lower)
    {
        if(uplo == Eigen::Lower)
        {
            for(Index j = 0; j < m_n; j++)
            {
                std::copy(&mat.coeffRef(j, j), &mat.coeffRef(0, j + 1), m_mat[j]);
            }
        } else {
            Scalar* dest = m_data.data();
            for(Index i = 0; i < m_n; i++)
            {
                for(Index j = i; j < m_n; j++, dest++)
                {
                    *dest = mat.coeff(i, j);
                }
            }
        }
    }

    void compress_permutation()
    {
        for(Index i = 0; i < m_n; i++)
        {
            const Index perm = std::abs(m_perm[i]);
            if(perm != i)
                m_permc.push_back(std::make_pair(i, perm));
        }
    }

    void print_mat() const
    {
        Matrix mat = Matrix::Zero(m_n, m_n);
        for(Index j = 0; j < m_n; j++)
        {
            std::copy(m_mat[j], m_mat[j] + (m_n - j), &mat.coeffRef(j, j));
        }
        std::cout << mat << std::endl << std::endl;
    }

    // Working on the A[k:end, k:end] submatrix
    // Exchange k <-> r
    // Assume r >= k
    void pivoting_1x1(Index k, Index r)
    {
        // No permutation
        if(k == r)
        {
            m_perm[k] = r;
            return;
        }

        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        // A[k, k] <-> A[r, r]
        std::swap(matp[k][0], matp[r][0]);

        // A[(r+1):end, k] <-> A[(r+1):end, r]
        std::swap_ranges(matp[k] + (r + 1 - k), matp[k + 1], matp[r] + 1);

        // A[(k+1):(r-1), k] <-> A[r, (k+1):(r-1)]
        Scalar* src = matp[k] + 1;
        for(Index j = k + 1; j < r; j++, src++)
        {
            std::swap(*src, matp[j][r - j]);
        }

        m_perm[k] = r;
    }

    // Working on the A[k:end, k:end] submatrix
    // Exchange [k+1, k] <-> [r, p]
    // Assume p >= k, r >= k+1
    void pivoting_2x2(Index k, Index r, Index p)
    {
        pivoting_1x1(k, p);
        pivoting_1x1(k + 1, r);

        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        // A[k+1, k] <-> A[r, k]
        std::swap(matp[k][1], matp[k][r - k]);

        // Use negative signs to indicate a 2x2 block
        m_perm[k] = -m_perm[k];
        m_perm[k + 1] = -m_perm[k + 1];
    }

    // A[r1, c1:c2] <-> A[r2, c1:c2]
    // Assume r2 >= r1 > c2 >= c1
    void interchange_rows(Index r1, Index r2, Index c1, Index c2)
    {
        if(r1 == r2)
            return;

        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        for(Index j = c1; j <= c2; j++)
        {
            std::swap(matp[j][r1 - j], matp[j][r2 - j]);
        }
    }

    // lambda = |A[r, k]| = max{|A[k+1, k]|, ..., |A[end, k]|}
    // Assume k < end
    Scalar find_lambda(Index k, Index& r)
    {
        using std::abs;

        const Scalar* head = m_mat[k];
        const Scalar* end = m_mat[k + 1];
        Scalar lambda = abs(head[1]);
        r = k + 1;
        for(const Scalar* ptr = head + 2; ptr < end; ptr++)
        {
            const Scalar abs_elem = abs(*ptr);
            if(lambda < abs_elem)
            {
                lambda = abs_elem;
                r = k + (ptr - head);
            }
        }
        return lambda;
    }

    // sigma = |A[p, r]| = max {|A[k, r]|, ..., |A[end, r]|} \ {A[r, r]}
    // Assume k < r < end
    Scalar find_sigma(Index k, Index r, Index& p)
    {
        using std::abs;

        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        // First search A[r+1, r], ...,  A[end, r], which has the same task as find_lambda()
        // If r == end, we skip this search
        Scalar sigma = Scalar(-1);
        if(r < m_n - 1)
            sigma = find_lambda(r, p);

        // Then search A[k, r], ..., A[r-1, r], which maps to A[r, k], ..., A[r, r-1]
        for(Index j = k; j < r; j++)
        {
            const Scalar abs_elem = abs(matp[j][r - j]);
            if(sigma < abs_elem)
            {
                sigma = abs_elem;
                p = j;
            }
        }

        return sigma;
    }

    // Generate permutations and apply to A
    // Return true if the resulting pivoting is 1x1, and false if 2x2
    bool permutate_mat(Index k, const Scalar& alpha)
    {
        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        Scalar* col_head = matp[k];
        Index r = k, p = k;
        const Scalar lambda = find_lambda(k, r);
        // std::cout << "lambda = " << lambda << std::endl;

        // If lambda=0, no need to interchange
        if(lambda > Scalar(0))
        {
            const Scalar abs_akk = abs(col_head[0]);
            // If |A[k, k]| >= alpha * lambda, no need to interchange
            if(abs_akk < alpha * lambda)
            {
                const Scalar sigma = find_sigma(k, r, p);
                // std::cout << "sigma = " << sigma << std::endl;

                // If sigma * |A[k, k]| >= alpha * lambda^2, no need to interchange
                if(sigma * abs_akk < alpha * lambda * lambda)
                {
                    if(abs_akk >= alpha * sigma)
                    {
                        // Permutation on A
                        pivoting_1x1(k, r);

                        // Permutation on L
                        interchange_rows(k, r, 0, k - 1);
                        return true;
                    } else {
                        // Permutation on A
                        // [r, p] and [p, r] are symmetric, but we need to make sure
                        // p >= k and r >= k+1, so it is safe to always make r > p
                        // One exception is when min{r,p} == k+1, in which case we make
                        // r = k+1, so that only one permutation needs to be performed
                        const Scalar rp_min = std::min(r, p);
                        const Scalar rp_max = std::max(r, p);
                        if(rp_min == k + 1)
                        {
                            r = rp_min; p = rp_max;
                        } else {
                            r = rp_max; p = rp_min;
                        }
                        pivoting_2x2(k, r, p);
                        // Permutation on L
                        interchange_rows(k, p, 0, k - 1);
                        interchange_rows(k + 1, r, 0, k - 1);
                        return false;
                    }
                }
            }
        }

        return true;
    }

    // E = [e11, e12]
    //     [e21, e22]
    // Overwrite E with inv(E)
    void inverse_inplace_2x2(Scalar& e11, Scalar& e21, Scalar& e22) const
    {
        // inv(E) = [d11, d12], d11 = e22/delta, d21 = -e21/delta, d22 = e11/delta
        //          [d21, d22]
        const Scalar delta = e11 * e22 - e21 * e21;
        std::swap(e11, e22);
        e11 /= delta;
        e22 /= delta;
        e21 = -e21 / delta;
    }

    void gaussian_elimination_1x1(Index k)
    {
        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        // D = 1 / A[k, k]
        const Scalar akk = matp[k][0];
        matp[k][0] = 1.0 / akk;

        // B -= l * l' / A[k, k], B := A[(k+1):end, (k+1):end], l := L[(k+1):end, k]
        Scalar* lptr = matp[k] + 1;
        const Index ldim = m_n - k - 1;
        MapVec l(lptr, ldim);
        for(Index j = 0; j < ldim; j++)
        {
            MapVec(matp[j + k + 1], ldim - j).noalias() -= (lptr[j] / akk) * l.tail(ldim - j);
        }

        // l /= A[k, k]
        l /= akk;
    }

    void gaussian_elimination_2x2(Index k)
    {
        // matp[j][i - j] -> A[i, j], i >= j
        Scalar** matp = &m_mat.front();

        // D = inv(E)
        Scalar& e11 = matp[k][0];
        Scalar& e21 = matp[k][1];
        Scalar& e22 = matp[k + 1][0];
        inverse_inplace_2x2(e11, e21, e22);

        // X = l * inv(E), l := L[(k+2):end, k:(k+1)]
        Scalar* l1ptr = matp[k] + 2;
        Scalar* l2ptr = matp[k + 1] + 1;
        const Index ldim = m_n - k - 2;
        MapVec l1(l1ptr, ldim), l2(l2ptr, ldim);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> X(ldim, 2);
        X.col(0).noalias() = l1 * e11 + l2 * e21;
        X.col(1).noalias() = l1 * e21 + l2 * e22;

        // B -= l * inv(E) * l' = X * l', B = A[(k+2):end, (k+2):end]
        for(Index j = 0; j < ldim; j++)
        {
            MapVec(matp[j + k + 2], ldim - j).noalias() -= (X.col(0).tail(ldim - j) * l1ptr[j] + X.col(1).tail(ldim - j) * l2ptr[j]);
        }

        // l = X
        l1.noalias() = X.col(0);
        l2.noalias() = X.col(1);
    }

public:
    BKLDLT() :
        m_n(0), m_computed(false)
    {}

    BKLDLT(ConstGenericMatrix& mat) :
        m_n(mat.rows()), m_computed(false)
    {
        compute(mat);
    }

    void compute(ConstGenericMatrix& mat, int uplo = Eigen::Lower)
    {
        using std::abs;

        m_n = mat.rows();
        if(m_n != mat.cols())
            throw std::invalid_argument("BKLDLT: matrix must be square");

        m_perm.setLinSpaced(m_n, 0, m_n - 1);
        m_permc.clear();

        // Copy data
        m_data.resize((m_n * (m_n + 1)) / 2);
        compute_pointer();
        copy_data(mat, uplo);

        const Scalar alpha = (1.0 + std::sqrt(17.0)) / 8.0;
        Index k = 0;
        for(k = 0; k < m_n - 1; k++)
        {
            // std::cout << "k = " << k << std::endl;

            // 1. Interchange rows and columns of A, and save the result to m_perm
            bool is_1x1 = permutate_mat(k, alpha);

            // 2. Gaussian elimination
            if(is_1x1)
            {
                gaussian_elimination_1x1(k);
            } else {
                gaussian_elimination_2x2(k);
                k++;
            }
        }
        // Invert the last 1x1 block if it exists
        if(k == m_n - 1)
        {
            m_mat[k][0] = Scalar(1) / m_mat[k][0];
        }

        compress_permutation();

        std::cout << "decomposition result:" << std::endl;
        print_mat();
        std::cout << "permutation result:" << std::endl;
        std::cout << m_perm.transpose() << std::endl << std::endl;
        std::cout << "compressed permutation result:" << std::endl;
        for(Index i = 0; i < m_permc.size(); i++)
            std::cout << "(" << m_permc[i].first << ", " << m_permc[i].second << ") ";
        std::cout << std::endl << std::endl;

        m_computed = true;
    }

    // Solve Ax=b
    Vector solve(ConstGenericVector& b) const
    {
        // PAP' = LDL'
        // 1. b -> Pb
        Vector res = b;
        Scalar* x = res.data();
        Index npermc = m_permc.size();
        for(Index i = 0; i < npermc; i++)
        {
            std::swap(x[m_permc[i].first], x[m_permc[i].second]);
        }

        // 2. Lz = Pb
        // matp[j][i - j] -> A[i, j], i >= j
        Scalar* const* matp = &m_mat.front();
        // If m_perm[end] < 0, then end with m_n - 3, otherwise end with m_n - 2
        const Index end = (m_perm[m_n - 1] < 0) ? (m_n - 3) : (m_n - 2);
        for(Index i = 0; i <= end; i++)
        {
            const Index b1size = m_n - i - 1;
            const Index b2size = b1size - 1;
            if(m_perm[i] >= 0)
            {
                MapConstVec l(matp[i] + 1, b1size);
                res.segment(i + 1, b1size).noalias() -= l * x[i];
            } else {
                MapConstVec l1(matp[i] + 2, b2size);
                MapConstVec l2(matp[i + 1] + 1, b2size);
                res.segment(i + 2, b2size).noalias() -= (l1 * x[i] + l2 * x[i + 1]);
                i++;
            }
        }

        // 3. Dw = z
        for(Index i = 0; i < m_n; i++)
        {
            const Scalar e11 = matp[i][0];
            if(m_perm[i] >= 0)
            {
                x[i] *= e11;
            } else {
                const Scalar e21 = matp[i][1], e22 = matp[i + 1][0];
                const Scalar wi = x[i] * e11 + x[i + 1] * e21;
                x[i + 1] = x[i] * e21 + x[i + 1] * e22;
                x[i] = wi;
                i++;
            }
        }

        // 4. L'y = w
        // If m_perm[end] < 0, then start with m_n - 3, otherwise start with m_n - 2
        Index i = (m_perm[m_n - 1] < 0) ? (m_n - 3) : (m_n - 2);
        for(; i >= 0; i--)
        {
            const Index ldim = m_n - i - 1;
            MapConstVec l(matp[i] + 1, ldim);
            x[i] -= res.segment(i + 1, ldim).dot(l);

            if(m_perm[i] < 0)
            {
                MapConstVec l2(matp[i - 1] + 2, ldim);
                x[i - 1] -= res.segment(i + 1, ldim).dot(l2);
                i--;
            }
        }

        // 5. x = P'y
        for(Index i = npermc - 1; i >= 0; i--)
        {
            std::swap(x[m_permc[i].first], x[m_permc[i].second]);
        }

        return res;
    }
};


} // namespace Spectra

#endif // BK_LDLT_H
