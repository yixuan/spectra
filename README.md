# <a href="https://spectralib.org"><img src="https://spectralib.org/img/logo.png" width="200px" /></a>

![Basic CI](https://github.com/yixuan/spectra/workflows/Basic%20CI/badge.svg) [![codecov](https://codecov.io/gh/yixuan/spectra/branch/master/graph/badge.svg)](https://codecov.io/gh/yixuan/spectra)

> **NOTE**: Spectra 1.0.0 was released in 2021-07-01, with a lot of
> API-breaking changes. Please see the [migration guide](MIGRATION.md)
> for a smooth transition to the new version.

> **NOTE**: If you are interested in the future development of Spectra, please join
> [this thread](https://github.com/yixuan/spectra/issues/92) to share your comments and suggestions.

[**Spectra**](https://spectralib.org) stands for **Sp**arse **E**igenvalue **C**omputation **T**oolkit
as a **R**edesigned **A**RPACK. It is a C++ library for large scale eigenvalue
problems, built on top of [Eigen](http://eigen.tuxfamily.org),
an open source linear algebra library.

**Spectra** is implemented as a header-only C++ library, whose only dependency,
**Eigen**, is also header-only. Hence **Spectra** can be easily embedded in
C++ projects that require calculating eigenvalues of large matrices.

## Relation to ARPACK

[ARPACK](https://www.arpack.org/) is a software package written in
FORTRAN for solving large scale eigenvalue problems. The development of
**Spectra** is much inspired by ARPACK, and as the full name indicates,
**Spectra** is a redesign of the ARPACK library using the C++ language.

In fact, **Spectra** is based on the algorithm described in the
[ARPACK Users' Guide](http://li.mit.edu/Archive/Activities/Archive/CourseWork/Ju_Li/MITCourses/18.335/Doc/ARPACK/Lehoucq97.pdf),
the implicitly restarted Arnoldi/Lanczos method. However,
it does not use the ARPACK code, and it is **NOT** a clone of ARPACK for C++.
In short, **Spectra** implements the major algorithms in ARPACK,
but **Spectra** provides a completely different interface, and it does not
depend on ARPACK.

## Common Usage

**Spectra** is designed to calculate a specified number (`k`) of eigenvalues
of a large square matrix (`A`). Usually `k` is much smaller than the size of the matrix
(`n`), so that only a few eigenvalues and eigenvectors are computed, which
in general is more efficient than calculating the whole spectral decomposition.
Users can choose eigenvalue selection rules to pick the eigenvalues of interest,
such as the largest `k` eigenvalues, or eigenvalues with largest real parts, etc.

To use the eigen solvers in this library, the user does not need to directly
provide the whole matrix, but instead, the algorithm only requires certain operations
defined on `A`. In the basic setting, it is simply the matrix-vector
multiplication. Therefore, if the matrix-vector product `A * x` can be computed
efficiently, which is the case when `A` is sparse, **Spectra**
will be very powerful for large scale eigenvalue problems.

There are two major steps to use the **Spectra** library:

1. Define a class that implements a certain matrix operation, for example the
matrix-vector multiplication `y = A * x` or the shift-solve operation
`y = inv(A - σ * I) * x`. **Spectra** has defined a number of
helper classes to quickly create such operations from a matrix object.
See the documentation of
[DenseGenMatProd](https://spectralib.org/doc/classSpectra_1_1DenseGenMatProd.html),
[DenseSymShiftSolve](https://spectralib.org/doc/classSpectra_1_1DenseSymShiftSolve.html), etc.
2. Create an object of one of the eigen solver classes, for example
[SymEigsSolver](https://spectralib.org/doc/classSpectra_1_1SymEigsSolver.html)
for symmetric matrices, and
[GenEigsSolver](https://spectralib.org/doc/classSpectra_1_1GenEigsSolver.html)
for general matrices. Member functions
of this object can then be called to conduct the computation and retrieve the
eigenvalues and/or eigenvectors.

Below is a list of the available eigen solvers in **Spectra**:

- [SymEigsSolver](https://spectralib.org/doc/classSpectra_1_1SymEigsSolver.html):
For real symmetric matrices
- [GenEigsSolver](https://spectralib.org/doc/classSpectra_1_1GenEigsSolver.html):
For general real- and complex-valued matrices
- [SymEigsShiftSolver](https://spectralib.org/doc/classSpectra_1_1SymEigsShiftSolver.html):
For real symmetric matrices using the shift-and-invert mode
- [GenEigsRealShiftSolver](https://spectralib.org/doc/classSpectra_1_1GenEigsRealShiftSolver.html):
For general real matrices using the shift-and-invert mode,
with a real-valued shift
- [GenEigsComplexShiftSolver](https://spectralib.org/doc/classSpectra_1_1GenEigsComplexShiftSolver.html):
For general real matrices using the shift-and-invert mode,
with a complex-valued shift
- [SymGEigsSolver](https://spectralib.org/doc/classSpectra_1_1SymGEigsSolver.html):
For generalized eigen solver with real symmetric matrices
- [SymGEigsShiftSolver](https://spectralib.org/doc/classSpectra_1_1SymGEigsShiftSolver.html):
For generalized eigen solver with real symmetric matrices, using the shift-and-invert mode
- [DavidsonSymEigsSolver](https://spectralib.org/doc/classSpectra_1_1DavidsonSymEigsSolver.html):
Jacobi-Davidson eigen solver for real symmetric matrices, with the DPR correction method
- [HermEigsSolver](https://spectralib.org/doc/classSpectra_1_1HermEigsSolver.html):
For complex Hermitian matrices

## Examples

Below is an example that demonstrates the use of the eigen solver for symmetric
matrices.

```cpp
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
// <Spectra/MatOp/DenseSymMatProd.h> is implicitly included
#include <iostream>

using namespace Spectra;

int main()
{
    // We are going to calculate the eigenvalues of M
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    Eigen::MatrixXd M = A + A.transpose();

    // Construct matrix operation object using the wrapper class DenseSymMatProd
    DenseSymMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(SortRule::LargestAlge);

    // Retrieve results
    Eigen::VectorXd evalues;
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n" << evalues << std::endl;

    return 0;
}
```

Sparse matrix is supported via classes such as `SparseGenMatProd`
and `SparseSymMatProd`.

```cpp
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <iostream>

using namespace Spectra;

int main()
{
    // A band matrix with 1 on the main diagonal, 2 on the below-main subdiagonal,
    // and 3 on the above-main subdiagonal
    const int n = 10;
    Eigen::SparseMatrix<double> M(n, n);
    M.reserve(Eigen::VectorXi::Constant(n, 3));
    for(int i = 0; i < n; i++)
    {
        M.insert(i, i) = 1.0;
        if(i > 0)
            M.insert(i - 1, i) = 3.0;
        if(i < n - 1)
            M.insert(i + 1, i) = 2.0;
    }

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn);

    // Retrieve results
    Eigen::VectorXcd evalues;
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n" << evalues << std::endl;

    return 0;
}
```

And here is an example for user-supplied matrix operation class.

```cpp
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <iostream>

using namespace Spectra;

// M = diag(1, 2, ..., 10)
class MyDiagonalTen
{
public:
    using Scalar = double;  // A typedef named "Scalar" is required
    int rows() const { return 10; }
    int cols() const { return 10; }
    // y_out = M * x_in
    void perform_op(const double *x_in, double *y_out) const
    {
        for(int i = 0; i < rows(); i++)
        {
            y_out[i] = x_in[i] * (i + 1);
        }
    }
};

int main()
{
    MyDiagonalTen op;
    SymEigsSolver<MyDiagonalTen> eigs(op, 3, 6);
    eigs.init();
    eigs.compute(SortRule::LargestAlge);
    if(eigs.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues();
        std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    }

    return 0;
}
```

## Shift-and-invert Mode

When it is needed to find eigenvalues that are closest to a number `σ`,
for example to find the smallest eigenvalues of a positive definite matrix
(in which case `σ = 0`), it is advised to use the shift-and-invert mode
of eigen solvers.

In the shift-and-invert mode, selection rules are applied to `1/(λ - σ)`
rather than `λ`, where `λ` are eigenvalues of `A`.
To use this mode, users need to define the shift-solve matrix operation. See
the documentation of
[SymEigsShiftSolver](https://spectralib.org/doc/classSpectra_1_1SymEigsShiftSolver.html)
for details.

## Complex-valued Matrices

**Spectra** provides the [HermEigsSolver](https://spectralib.org/doc/classSpectra_1_1HermEigsSolver.html) solver for complex-valued Hermitian matrices,
and the [GenEigsSolver](https://spectralib.org/doc/classSpectra_1_1GenEigsSolver.html) solver for general complex-valued matrices. See the example below.

```cpp
#include <Eigen/Core>
#include <Spectra/HermEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <iostream>

using namespace Spectra;

int main()
{
    std::srand(0);

    // We are going to calculate the eigenvalues of H and G
    Eigen::MatrixXcd G = Eigen::MatrixXcd::Random(10, 10);
    // H is Hermitian
    Eigen::MatrixXcd H = G + G.adjoint();

    // Construct matrix operation objects using the wrapper
    // classes DenseHermMatProd and DenseGenMatProd
    using OpHType = DenseHermMatProd<std::complex<double>>;
    using OpGType = DenseGenMatProd<std::complex<double>>;
    OpHType opH(H);
    OpGType opG(G);

    // Construct solver object for H, requesting the largest three eigenvalues
    HermEigsSolver<OpHType> eigsH(opH, 3, 6);

    // Initialize and compute
    eigsH.init();
    int nconvH = eigsH.compute(SortRule::LargestAlge);

    // Retrieve results
    // Eigenvalues are real-valued, and eigenvectors are complex-valued
    if (eigsH.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evaluesH = eigsH.eigenvalues();
        std::cout << "Eigenvalues of H found:\n" << evaluesH << std::endl;
        Eigen::MatrixXcd evecsH = eigsH.eigenvectors();
        std::cout << "Eigenvectors of H:\n" << evecsH << std::endl;
    }

    // Similar procedure for matrix G
    GenEigsSolver<OpGType> eigsG(opG, 3, 6);
    eigsG.init();
    int nconvG = eigsG.compute(SortRule::LargestMagn);
    if (eigsG.info() == CompInfo::Successful)
    {
        Eigen::VectorXcd evaluesG = eigsG.eigenvalues();
        std::cout << "Eigenvalues of G found:\n" << evaluesG << std::endl;
        Eigen::MatrixXcd evecsG = eigsG.eigenvectors();
        std::cout << "Eigenvectors of G:\n" << evecsG << std::endl;
    }

    return 0;
}
```

## Documentation

The [API reference](https://spectralib.org/doc/) page contains the documentation
of **Spectra** generated by [Doxygen](http://www.doxygen.org/),
including all the background knowledge, example code and class APIs.

More information can be found in the project page [https://spectralib.org](https://spectralib.org).

## Installation

An optional CMake installation is supported, if you have CMake with at least v3.10 installed. You can install the headers using the following commands:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX='intended installation directory' -DBUILD_TESTS=TRUE
make all && make test && make install
```

By installing **Spectra** in this way, you also create a CMake target `Spectra::Spectra` that can be used in subsequent build procedures for other programs.

If you already have Eigen installed, you can specify the installation directory by setting the `CMAKE_PREFIX_PATH` variable or `Eigen3_ROOT`.
For example:

```bash
cmake .. -DCMAKE_INSTALL_PREFIX='intended installation directory' -DCMAKE_PREFIX_PATH='path where the installation of Eigen3 can be found' -DBUILD_TESTS=TRUE
```

A couple of useful environment variables can be set to control the download of Eigen. `CPM_DOWNLOAD_ALL=ON` will force the download of Eigen, even if an installation is already present on the system. `CPM_LOCAL_PACKAGES_ONLY=ON` will force the opposite behavior. The download directory can be controlled by setting the variable `CPM_SOURCE_CACHE`.

## License

**Spectra** is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/), the same license used by **Eigen**.
