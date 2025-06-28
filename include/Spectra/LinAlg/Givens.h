// Copyright (C) 2025 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPECTRA_GIVENS_H
#define SPECTRA_GIVENS_H

#include <Eigen/Core>
#include <cmath>    // std::sqrt, std::pow, std::hypot
#include <complex>  // std::complex

#include "../Util/TypeTraits.h"

/// \cond

namespace Spectra {

// Helper class to compute Givens rotations in a numerically stable way
// Here Scalar is a real type, e.g., double
template <typename Scalar>
struct StableScaling
{
    using Complex = std::complex<Scalar>;

    // Given a >= b > 0, compute r = sqrt(a^2 + b^2), c = a / r, and s = b / r with a high precision
    static void run(const Scalar& a, const Scalar& b, Scalar& r, Scalar& c, Scalar& s)
    {
        using std::pow;
        using std::hypot;

        // Let t = b / a, then 0 < t <= 1
        // c = 1 / sqrt(1 + t^2)
        // s = t * c
        // r = a * sqrt(1 + t^2)
        const Scalar t = b / a;
        // We choose a cutoff such that cutoff^4 < eps
        // If t >= cutoff, use the standard way; otherwise use Taylor series expansion
        // to avoid an explicit sqrt() call that may lose precision
        const Scalar eps = TypeTraits<Scalar>::epsilon();
        // std::pow() is not constexpr, so we do not declare cutoff to be constexpr
        // But most compilers should be able to compute cutoff at compile time
        // cutoff ~= 1.22e-5
        const Scalar cutoff = Scalar(0.1) * pow(eps, Scalar(0.25));
        if (t >= cutoff)
        {
            // When t is not too small, we can directly compute r, c, s
            r = hypot(a, b);
            c = a / r;
            s = b / r;
        }
        else
        {
            // 1 / sqrt(1 + t^2) ~=     1 - (1/2) * t^2 + (3/8) * t^4 - (5/16) * t^6
            // 1 / sqrt(1 + l^2) ~= 1 / l - (1/2) / l^3 + (3/8) / l^5 - (5/16) / l^7, where l = 1 / t
            // 1 / sqrt(1 + l^2) == t / sqrt(1 + t^2)
            //                   ~=     t - (1/2) * t^3 + (3/8) * t^5 - (5/16) * t^7
            // sqrt(1 + t^2)     ~=     1 + (1/2) * t^2 - (1/8) * t^4 + (1/16) * t^6
            //
            // c = 1 / sqrt(1 + t^2) ~= 1 - t^2 * (1/2 - (3/8) * t^2 + (5/16) * t^4)
            //                       == 1 - t^2 * (1/2 - t^2 * (3/8 - (5/16) * t^2))
            // s = 1 / sqrt(1 + l^2) ~= t * (1 - t^2 * (1/2 - t^2 *(3/8 - (5/16) * t^2)))
            // r = a * sqrt(1 + t^2) ~= a + (1/2) * b * t - (1/8) * b * t^3 + (1/16) * b * t^5
            //                       == a + (b/2) * t * (1 - t^2 * (1/4 - 1/8 * t^2))
            const Scalar c1 = Scalar(1);
            const Scalar c2 = Scalar(0.5);
            const Scalar c4 = Scalar(0.25);
            const Scalar c8 = Scalar(0.125);
            const Scalar c38 = Scalar(0.375);
            const Scalar c516 = Scalar(0.3125);
            const Scalar t2 = t * t;
            c = c1 - t2 * (c2 - t2 * (c38 - c516 * t2));
            s = t * c;
            r = a + c2 * b * t * (c1 - t2 * (c4 - c8 * t2));

            /* const Scalar t_2 = Scalar(0.5) * t;
            const Scalar t2_2 = t_2 * t;
            const Scalar t3_2 = t2_2 * t;
            const Scalar t4_38 = Scalar(1.5) * t2_2 * t2_2;
            const Scalar t5_16 = Scalar(0.25) * t3_2 * t2_2;
            c = Scalar(1) - t2_2 + t4_38;
            s = t - t3_2 + Scalar(6) * t5_16;
            r = a + b * (t_2 - Scalar(0.25) * t3_2 + t5_16); */
        }
    }

    // Given |a|_1 >= |b|_1 > 0,
    // compute a2 = |a|^2, t = |b| / |a|, tc1 = sqrt(1 + t^2), tc2 = 1 / sqrt(1 + t^2)
    static void run(const Complex& a, const Complex& b, Scalar& a2, Scalar& tc1, Scalar& tc2)
    {
        using std::sqrt;

        const Scalar b2 = Eigen::numext::abs2(b);
        a2 = Eigen::numext::abs2(a);
        // |x| <= |x|_1 <= sqrt(2) * |x|
        // 0 < |b| / |a| <= sqrt(2) * |b|_1 / |a|_1 <= sqrt(2)
        // 0 < t2 <= 2
        const Scalar t2 = b2 / a2;

        // We choose a cutoff such that cutoff^2 < eps
        // If t2 >= cutoff, use the standard way; otherwise use Taylor series expansion
        // to avoid an explicit sqrt() call that may lose precision
        const Scalar eps = TypeTraits<Scalar>::epsilon();
        const Scalar cutoff = Scalar(0.1) * sqrt(eps);
        if (t2 >= cutoff)
        {
            tc1 = sqrt(Scalar(1) + t2);
            tc2 = sqrt(a2 / (a2 + b2));
        }
        else
        {
            // sqrt(1 + t^2)     ~= 1 + (1/2) * t^2 - (1/8) * t^4 + (1/16) * t^6
            //                   == 1 + t^2 * (1/2 - (1/8) * t^2 + (1/16) * t^4)
            //                   == 1 + t^2 * (1/2 - t^2 * (1/8 - (1/16) * t^2))
            // 1 / sqrt(1 + t^2) ~= 1 - (1/2) * t^2 + (3/8) * t^4 - (5/16) * t^6
            //                   == 1 - t^2 * (1/2 - t^2 * (3/8 - (5/16) * t^2))
            const Scalar c1 = Scalar(1);
            const Scalar c2 = Scalar(0.5);
            const Scalar c8 = Scalar(0.125);
            const Scalar c16 = Scalar(0.0625);
            const Scalar c38 = Scalar(0.375);
            const Scalar c516 = Scalar(0.3125);
            tc1 = c1 + t2 * (c2 - t2 * (c8 - c16 * t2));
            tc2 = c1 - t2 * (c2 - t2 * (c38 - c516 * t2));
        }
    }
};

// Consider the rotation matrix
//     G = [  c   s]
//         [-(s)  c]
// where c is real, and (s) is the conjugate of s
//
// G is responsible for transforming a 2x1 vector u = [x] into v = [r]:
//                                                    [y]          [0]
// G^H u = v  ==>  [ c   -s] [x] = [r]  ==>   c  * x - s * y = r
//                 [(s)   c] [y] = [0]       (s) * x + c * y = 0
//
// When x and y are real, choose r = sqrt(x^2 + y^2), c = x / r, s = -y / r
// When x and y are complex, let
// rho = sqrt(|x|^2 + |y|^2), z = { x / |x|, if x != 0
//                                { 1,       if x = 0
// r = z * rho, c = x / r = |x| / rho, s = -(y) / (r) = -z * (y) / rho
//
// Reference: https://www.netlib.org/lapack/lawnspdf/lawn150.pdf

// Default implementation for real type
template <typename Scalar>
class Givens
{
private:
    // The type of the real part, e.g.,
    //     Scalar = double               => RealScalar = double
    //     Scalar = std::complex<double> => RealScalar = double
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

public:
    // c is always real, and other variables can be real or complex
    //
    // Given x and y, compute 1) r = sqrt(x^2 + y^2), 2) c = x / r, 3) s = -y / r
    // If both x and y are zero, set c = 1 and s = 0
    // We must implement it in a numerically stable way
    // The implementation below is shown to be more accurate than directly computing
    //     r = std::hypot(x, y); c = x / r; s = -y / r;
    static void compute_rotation(const Scalar& x, const Scalar& y, Scalar& r, RealScalar& c, Scalar& s)
    {
        using std::abs;

        // Only need xsign when x != 0
        const Scalar xsign = (x > Scalar(0)) ? Scalar(1) : Scalar(-1);
        const Scalar xabs = abs(x);
        if (y == Scalar(0))
        {
            c = (x == Scalar(0)) ? Scalar(1) : xsign;
            s = Scalar(0);
            r = xabs;
            return;
        }

        // Now we know y != 0
        const Scalar ysign = (y > Scalar(0)) ? Scalar(1) : Scalar(-1);
        const Scalar yabs = abs(y);
        if (x == Scalar(0))
        {
            c = Scalar(0);
            s = -ysign;
            r = yabs;
            return;
        }

        // Now we know x != 0, y != 0
        if (xabs >= yabs)
        {
            StableScaling<Scalar>::run(xabs, yabs, r, c, s);
            c = xsign * c;
            s = -ysign * s;
        }
        else
        {
            StableScaling<Scalar>::run(yabs, xabs, r, s, c);
            c = xsign * c;
            s = -ysign * s;
        }
    }
};

// Specialization for complex values
template <typename RealScalar>
class Givens<std::complex<RealScalar>>
{
private:
    using Scalar = std::complex<RealScalar>;
    using Complex = Scalar;

public:
    // c is real, and other variables are complex
    static void compute_rotation(const Scalar& x, const Scalar& y, Scalar& r, RealScalar& c, Scalar& s)
    {
        using std::sqrt;

        const Complex zero(RealScalar(0), RealScalar(0));
        if (y == zero)
        {
            c = RealScalar(1);
            s = zero;
            r = x;
            return;
        }

        // Now we know y != 0
        if (x == zero)
        {
            // x = 0, rho = |y|, z = 1, r = rho, c = 0, s = -(y) / |y|
            // Assume that y = a + bi, then r = sqrt(a^2 + b^2),
            // s = (-a + bi) / r = -(a / r) + (b / r)i
            // So it is equivalent to computing the Givens rotation for (-a, -b)
            c = RealScalar(0);
            const RealScalar yr = Eigen::numext::real(y), yi = Eigen::numext::imag(y);
            RealScalar rr, sr, si;
            Givens<RealScalar>::compute_rotation(-yr, -yi, rr, sr, si);
            s = Complex(sr, si);
            r = Complex(rr, RealScalar(0));
            return;
        }

        // Now we know x != 0, y != 0
        // l1-norm of x and y
        const RealScalar xnorm1 = Eigen::numext::norm1(x);
        const RealScalar ynorm1 = Eigen::numext::norm1(y);
        if (xnorm1 > ynorm1)
        {
            // |x|_1 > |y|_1
            //
            // Algorithm 1
            // t = |y| / |x|, rho = sqrt(|x|^2 + |y|^2) = |x| * sqrt(1 + t^2)
            // c = |x| / rho = 1 / sqrt(1 + t^2)
            // r = z * rho = (x / |x|) * rho = x * sqrt(1 + t^2)
            // s = -z * (y) / rho = -x * (y) / (|x| * |x| * sqrt(1 + t^2))
            //   = -c * x * (y) / |x|^2
            RealScalar x2, tc1, tc2;
            StableScaling<RealScalar>::run(x, y, x2, tc1, tc2);
            c = tc2;
            r = tc1 * x;
            s = -(c / x2) * (x * Eigen::numext::conj(y));

            /*
            // Algorithm 2
            // xs = x / |x|_1, ys = y / |x|_1, t = |ys| / |xs|
            // s = -c * xs * (ys) * (|x|_1 / |x|)^2 = -c * xs * (ys) / |xs|^2
            const Complex xs = x / xnorm1, ys = y / xnorm1;
            RealScalar xs2, tc1, tc2;
            StableScaling<RealScalar>::run(xs, ys, xs2, tc1, tc2);
            c = tc2;
            r = tc1 * x;
            s = -(c / xs2) * (xs * Eigen::numext::conj(ys));
            */
        }
        else
        {
            // |x|_1 < |y|_1
            //
            // Algorithm 1
            // xs = x / |y|_1, ys = y / |y|_1
            // rho = sqrt(|x|^2 + |y|^2) = |y|_1 * sqrt(|xs|^2 + |ys|^2)
            // z = x / |x|, r = z * rho, c = |x| / rho
            // s = -z * (y) / rho

            // const Complex xs = x / ynorm1, ys = y / ynorm1;
            // const RealScalar rho = ynorm1 * sqrt(Eigen::numext::abs2(xs) + Eigen::numext::abs2(ys));
            const RealScalar rho = sqrt(Eigen::numext::abs2(x) + Eigen::numext::abs2(y));
            // Scale x, equivalent to Givens rotation on (x.real, -x.imag)
            const RealScalar xr = Eigen::numext::real(x), xi = Eigen::numext::imag(x);
            RealScalar xnorm, zr, zi;
            Givens<RealScalar>::compute_rotation(xr, -xi, xnorm, zr, zi);
            const Complex z(zr, zi);
            r = rho * z;
            c = xnorm / rho;
            s = -(z * Eigen::numext::conj(y)) / rho;

            /*
            // Algorithm 2
            // z = x / |x|, w = y / |y|
            // rho = sqrt(|x|^2 + |y|^2), r = z * rho, c = |x| / rho
            // s = -z * (y) / rho = -z * (w) * |y| / rho

            // Scale x, equivalent to Givens rotation on (x.real, -x.imag)
            const RealScalar xr = Eigen::numext::real(x), xi = Eigen::numext::imag(x);
            RealScalar xnorm, zr, zi;
            Givens<RealScalar>::compute_rotation(xr, -xi, xnorm, zr, zi);

            // Scale y, equivalent to Givens rotation on (y.real, -y.imag)
            const RealScalar yr = Eigen::numext::real(y), yi = Eigen::numext::imag(y);
            RealScalar ynorm, wr, wi;
            Givens<RealScalar>::compute_rotation(yr, -yi, ynorm, wr, wi);

            // Compute rho = sqrt(|x|^2 + |y|^2), |x| / rho, and |y| / rho
            RealScalar rho, xrho, yrho;
            if (xnorm >= ynorm)
            {
                StableScaling<RealScalar>::run(xnorm, ynorm, rho, xrho, yrho);
            }
            else
            {
                StableScaling<RealScalar>::run(ynorm, xnorm, rho, yrho, xrho);
            }

            r = rho * Complex(zr, zi);
            c = xrho;
            // z * (w) = (z.real + z.imag * i) * (w.real - w.imag * i)
            //         = (z.real * w.real + z.imag * w.imag) + (z.imag * w.real - z.real * w.imag) * i
            s = -yrho * Complex(zr * wr + zi * wi, zi * wr - zr * wi);
            */
        }
    }
};

}  // namespace Spectra

/// \endcond

#endif  // SPECTRA_GIVENS_H
