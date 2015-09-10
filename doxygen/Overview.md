**ARPACK-Eigen** is a redesign of the [ARPACK](http://www.caam.rice.edu/software/ARPACK/)
software for large scale eigenvalue problems, built on top of
[Eigen](http://eigen.tuxfamily.org), an open source C++ linear algebra library.

**ARPACK-Eigen** is implemented as a header-only C++ library, whose only dependency,
**Eigen**, is also header-only. Hence **ARPACK-Eigen** can be easily embedded in
C++ projects that require solving large scale eigenvalue problems.

## Common Usage

**ARPACK-Eigen** is designed to calculate a specified number (\f$k\f$) of eigenvalues
of a large square matrix (\f$A\f$). Usually \f$k\f$ is much less than the size of matrix
(\f$n\f$), so that only a few eigenvalues and eigenvectors are computed, which
in general is more efficient than calculating the whole spectral decomposition.
Users can choose eigenvalue selection rules to pick up the eigenvalues of interest,
such as the largest \f$k\f$ eigenvalues, or eigenvalues with largest real parts,
etc.

To use the eigen solvers in this library, the user does not need to directly
provide the whole matrix, but instead, the algorithm only requires certain operations
defined on \f$A\f$, and in the basic setting, it is simply the matrix-vector
multiplication. Therefore, if the matrix-vector product \f$Ax\f$ can be computed
efficiently, which is the case when \f$A\f$ is sparse, **ARPACK-Eigen**
will be very powerful for large scale eigenvalue problems.

There are two major steps to use the **ARPACK-Eigen** library:

1. Define a class that implements a certain matrix operation, for example the
matrix-vector multiplication \f$y=Ax\f$ or the shift-solve operation
\f$y=(A-\sigma I)^{-1}x\f$. **ARPACK-Eigen** has defined a number of
helper classes to quickly create such operations from a matrix object.
See the documentation of DenseGenMatProd, DenseSymShiftSolve, etc.
2. Create an object of one of the eigen solver classes, for example SymEigsSolver
for symmetric matrices, and GenEigsSolver for general matrices. Member functions
of this object can then be called to conduct the computation and retrieve the
eigenvalues and/or eigenvectors.

Below is a list of the available eigen solvers in **ARPACK-Eigen**:
- SymEigsSolver: for real symmetric matrices
- GenEigsSolver: for general real matrices
- SymEigsShiftSolver: for real symmetric matrices using the shift-and-invert mode
- GenEigsRealShiftSolver: for general real matrices using the shift-and-invert mode,
with a real-valued shift

## Examples

Below is an example that demonstrates the use of the eigen solver for symmetric
matrices.

~~~~~~~~~~{.cpp}
#include <Eigen/Core>
#include <SymEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <iostream>

int main()
{
    // We are going to calculate the eigenvalues of M
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    Eigen::MatrixXd M = A + A.transpose();

    // Construct matrix operation object using the wrapper class DenseGenMatProd
    DenseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver< double, LARGEST_ALGE, DenseGenMatProd<double> > eigs(&op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    Eigen::VectorXd evalues;
    if(nconv > 0)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n" << evalues << std::endl;

    return 0;
}
~~~~~~~~~~

And here is an example for user-supplied matrix operation class.

~~~~~~~~~~{.cpp}
#include <Eigen/Core>
#include <SymEigsSolver.h>
#include <iostream>

// M = diag(1, 2, ..., 10)
class MyDiagonalTen
{
public:
    int rows() { return 10; }
    int cols() { return 10; }
    // y_out = M * x_in
    void perform_op(double *x_in, double *y_out)
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
    SymEigsSolver<double, LARGEST_ALGE, MyDiagonalTen> eigs(&op, 3, 6);
    eigs.init();
    eigs.compute();
    Eigen::VectorXd evalues = eigs.eigenvalues();
    std::cout << "Eigenvalues found:\n" << evalues << std::endl;

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
the documentation of SymEigsShiftSolver for details.

## License

**ARPACK-Eigen** is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/), the same license used by **Eigen**.
