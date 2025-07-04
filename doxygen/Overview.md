**Spectra** stands for <strong>Sp</strong>arse <strong>E</strong>igenvalue
<strong>C</strong>omputation <strong>T</strong>oolkit
as a <strong>R</strong>edesigned <strong>A</strong>RPACK.
It is a C++ library for large scale eigenvalue
problems, built on top of [Eigen](http://eigen.tuxfamily.org),
an open source linear algebra library.

**Spectra** is implemented as a header-only C++ library, whose only dependency,
**Eigen**, is also header-only. Hence **Spectra** can be easily embedded in
C++ projects that require calculating eigenvalues of large matrices.

The development page of **Spectra** is at
[https://github.com/yixuan/spectra/](https://github.com/yixuan/spectra).

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

**Spectra** is designed to calculate a specified number (\f$k\f$) of eigenvalues
of a large square matrix (\f$A\f$). Usually \f$k\f$ is much smaller than the size of the matrix
(\f$n\f$), so that only a few eigenvalues and eigenvectors are computed, which
in general is more efficient than calculating the whole spectral decomposition.
Users can choose eigenvalue selection rules to pick the eigenvalues of interest,
such as the largest \f$k\f$ eigenvalues, or eigenvalues with largest real parts, etc.

To use the eigen solvers in this library, the user does not need to directly
provide the whole matrix, but instead, the algorithm only requires certain operations
defined on \f$A\f$. In the basic setting, it is simply the matrix-vector
multiplication. Therefore, if the matrix-vector product \f$Ax\f$ can be computed
efficiently, which is the case when \f$A\f$ is sparse, **Spectra**
will be very powerful for large scale eigenvalue problems.

There are two major steps to use the **Spectra** library:

1. Define a class that implements a certain matrix operation, for example the
matrix-vector multiplication \f$y=Ax\f$ or the shift-solve operation
\f$y=(A-\sigma I)^{-1}x\f$. **Spectra** has defined a number of
helper classes to quickly create such operations from a matrix object.
See the documentation of \link Spectra::DenseGenMatProd DenseGenMatProd\endlink,
\link Spectra::DenseSymShiftSolve DenseSymShiftSolve\endlink, etc.
2. Create an object of one of the eigen solver classes, for example
\link Spectra::SymEigsSolver SymEigsSolver\endlink for symmetric matrices, and
\link Spectra::GenEigsSolver GenEigsSolver\endlink
for general matrices. Member functions of this object can then be called to
conduct the computation and retrieve the eigenvalues and/or eigenvectors.

Below is a list of the available eigen solvers in **Spectra**:
- \link Spectra::SymEigsSolver SymEigsSolver\endlink:
  For real symmetric matrices
- \link Spectra::GenEigsSolver GenEigsSolver\endlink:
  For general real- and complex-valued matrices
- \link Spectra::SymEigsShiftSolver SymEigsShiftSolver\endlink:
  For real symmetric matrices using the shift-and-invert mode
- \link Spectra::GenEigsRealShiftSolver GenEigsRealShiftSolver\endlink:
  For general real matrices using the shift-and-invert mode, with a real-valued shift
- \link Spectra::GenEigsComplexShiftSolver GenEigsComplexShiftSolver\endlink:
  For general real matrices using the shift-and-invert mode, with a complex-valued shift
- \link Spectra::SymGEigsSolver SymGEigsSolver\endlink:
  For generalized eigen solver with real symmetric matrices
- \link Spectra::SymGEigsShiftSolver SymGEigsShiftSolver\endlink:
  For generalized eigen solver with real symmetric matrices, using the shift-and-invert mode
- \link Spectra::DavidsonSymEigsSolver DavidsonSymEigsSolver\endlink:
  Jacobi-Davidson eigen solver for real symmetric matrices, with the DPR correction method
- \link Spectra::HermEigsSolver HermEigsSolver\endlink:
  For complex Hermitian matrices

## Examples

Below is an example that demonstrates the use of the eigen solver for symmetric
matrices.

~~~~~~~~~~{.cpp}
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
~~~~~~~~~~

Sparse matrix is supported via classes such as \link Spectra::SparseGenMatProd SparseGenMatProd\endlink
and \link Spectra::SparseSymMatProd SparseSymMatProd\endlink.

~~~~~~~~~~{.cpp}
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
~~~~~~~~~~

And here is an example for user-supplied matrix operation class.

~~~~~~~~~~{.cpp}
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
~~~~~~~~~~

## Shift-and-invert Mode

When it is needed to find eigenvalues that are closest to a number \f$\sigma\f$,
for example to find the smallest eigenvalues of a positive definite matrix
(in which case \f$\sigma=0\f$), it is advised to use the shift-and-invert mode
of eigen solvers.

In the shift-and-invert mode, selection rules are applied to \f$1/(\lambda-\sigma)\f$
rather than \f$\lambda\f$, where \f$\lambda\f$ are eigenvalues of \f$A\f$.
To use this mode, users need to define the shift-solve matrix operation. See
the documentation of \link Spectra::SymEigsShiftSolver SymEigsShiftSolver\endlink for details.

## Complex-valued Matrices

**Spectra** provides the \link Spectra::HermEigsSolver HermEigsSolver\endlink
solver for complex-valued Hermitian matrices,
and the \link Spectra::GenEigsSolver GenEigsSolver\endlink
solver for general complex-valued matrices. See the example below.

~~~~~~~~~~{.cpp}
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
~~~~~~~~~~

## License

**Spectra** is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/), the same license used by **Eigen**.
