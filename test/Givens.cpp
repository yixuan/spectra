// Test ../include/Spectra/LinAlg/Givens.h
#include <complex>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Jacobi>
#include <Spectra/LinAlg/Givens.h>

#include "catch.hpp"

using Vector = Eigen::VectorXd;
using Complex = std::complex<double>;
using ComplexVector = Eigen::VectorXcd;

// Fill x with random real numbers
void fill_vector(Vector& x, std::default_random_engine& gen)
{
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_real_distribution<double> distr(-100.0, 100.0);
    const int n = x.size();

    for (int i = 0; i < n; i++)
    {
        // With probability 0.1, set value to be zero
        x[i] = (unif(gen) < 0.1) ? 0.0 : distr(gen);
    }
}

// Fill x with random complex numbers
void fill_vector(ComplexVector& x, std::default_random_engine& gen)
{
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_real_distribution<double> distr(-100.0, 100.0);
    const int n = x.size();

    for (int i = 0; i < n; i++)
    {
        // With probability 0.1, set value to be zero
        double xr = (unif(gen) < 0.1) ? 0.0 : distr(gen);
        double xi = (unif(gen) < 0.1) ? 0.0 : distr(gen);
        x[i] = Complex(xr, xi);
    }
}

// Summarize results
void report_stats(const std::string& prefix, const Vector& r_err, const Vector& zero_err)
{
    const double r_log10err_mean = (r_err.array().abs() + 1e-32).log10().mean();
    const double r_err_max = r_err.array().abs().maxCoeff();
    const double zero_log10err_mean = (zero_err.array().abs() + 1e-32).log10().mean();
    const double zero_err_max = zero_err.array().abs().maxCoeff();
    INFO(prefix + " r_log10err_mean = " << r_log10err_mean);
    INFO(prefix + " r_err_max = " << r_err_max);
    REQUIRE(r_err_max == Approx(0.0).margin(1e-12));

    INFO(prefix + " zero_log10err_mean = " << zero_log10err_mean);
    INFO(prefix + " zero_err_max = " << zero_err_max);
    REQUIRE(zero_err_max == Approx(0.0).margin(1e-12));
}

TEST_CASE("Givens rotation on real numbers", "[Givens]")
{
    // Random number generation
    std::default_random_engine gen;
    gen.seed(0);
    const int nsim = 100000;
    Vector x(nsim), y(nsim);
    fill_vector(x, gen);
    fill_vector(y, gen);

    // Simulations
    Vector r_err_eigen(nsim), zero_err_eigen(nsim), r_err_spectra(nsim), zero_err_spectra(nsim);
    for (int i = 0; i < nsim; i++)
    {
        // Eigen implementation
        Eigen::JacobiRotation<double> G;
        double r_eigen;
        G.makeGivens(x[i], y[i], &r_eigen);
        // G = [ c  s], G' = [c  -s]
        //     [-s  c]     = [s   c]
        const double c_eigen = G.c(), s_eigen = G.s();
        r_err_eigen[i] = c_eigen * x[i] - s_eigen * y[i] - r_eigen;
        zero_err_eigen[i] = s_eigen * x[i] + c_eigen * y[i];

        // Spectra implementation
        double r_spectra, c_spectra, s_spectra;
        Spectra::Givens<double>::compute_rotation(x[i], y[i], r_spectra, c_spectra, s_spectra);
        // G = [ c  s], G' = [c  -s]
        //     [-s  c]     = [s   c]
        r_err_spectra[i] = c_spectra * x[i] - s_spectra * y[i] - r_spectra;
        zero_err_spectra[i] = s_spectra * x[i] + c_spectra * y[i];
    }

    report_stats("[Eigen]", r_err_eigen, zero_err_eigen);
    report_stats("[Spectra]", r_err_spectra, zero_err_spectra);
}

TEST_CASE("Givens rotation on complex numbers", "[Givens]")
{
    using std::abs;
    using std::conj;

    // Random number generation
    std::default_random_engine gen;
    gen.seed(0);
    const int nsim = 100000;
    ComplexVector x(nsim), y(nsim);
    fill_vector(x, gen);
    fill_vector(y, gen);

    // Simulations
    Vector r_err_eigen(nsim), zero_err_eigen(nsim), r_err_spectra(nsim), zero_err_spectra(nsim);
    for (int i = 0; i < nsim; i++)
    {
        // Eigen implementation
        Eigen::JacobiRotation<Complex> G;
        Complex r_eigen;
        G.makeGivens(x[i], y[i], &r_eigen);
        // G = [ c  (s)], G^H = [(c) -(s)]
        //     [-s  (c)]      = [ s    c ]
        const Complex c_eigen = G.c(), s_eigen = G.s();
        r_err_eigen[i] = abs(conj(c_eigen) * x[i] - conj(s_eigen) * y[i] - r_eigen);
        zero_err_eigen[i] = abs(s_eigen * x[i] + c_eigen * y[i]);

        // Spectra implementation
        double c_spectra;
        Complex r_spectra, s_spectra;
        // G = [  c   s], G^H = [ c  -s]
        //     [-(s)  c]      = [(s)  c]
        Spectra::Givens<Complex>::compute_rotation(x[i], y[i], r_spectra, c_spectra, s_spectra);
        r_err_spectra[i] = abs(c_spectra * x[i] - s_spectra * y[i] - r_spectra);
        zero_err_spectra[i] = abs(conj(s_spectra) * x[i] + c_spectra * y[i]);
    }

    report_stats("[Eigen]", r_err_eigen, zero_err_eigen);
    report_stats("[Spectra]", r_err_spectra, zero_err_spectra);

    // For debugging
    Eigen::Index i;
    zero_err_eigen.maxCoeff(&i);
    std::cout << "i = " << i << std::endl;
    std::cout << std::setprecision(18);
    std::cout << "x = " << x[i] << std::endl;
    std::cout << "y = " << y[i] << std::endl;
    zero_err_spectra.maxCoeff(&i);
    std::cout << "i = " << i << std::endl;
    std::cout << "x = " << x[i] << std::endl;
    std::cout << "y = " << y[i] << std::endl;
}
