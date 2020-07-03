This is an example of how to use the Jacobi-Davidson Symmetric Eigenvalue Solver with DPR correction method. This test can also be found as a full file in the [test/DavidsonSymEigen_example.ccp](test/JDSymEigsDPR_example.cpp) file and can be compiled with cmake and run afterwards

```bash
mkdir build && cd build && cmake ../
make JDSymEigsDPR_example
./test/JDSymEigsDPR_example
```

Suppose we want to find the 2 eigenpairs with the Largest value from a 1000x1000 Sparse Matrix A, then we could use this solver to quickly find them.


- First we have to construct the matrix

`Note: The Matrix has to be diagonally dominant otherwise the method will not converge`

```cpp
#include <Eigen/Sparse>
#include <random>

// Generate Matrix
Eigen::SparseMatrix<double> gen_sym_data_sparse(int n)
{
    double prob = 0.5;
    Eigen::SparseMatrix<double> mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = distr(gen) - 0.5;
            if (i == j)
                mat.coeffRef(i, j) = 10 * n;
        }
    }
    return mat + Eigen::SparseMatrix<double>(mat.transpose());
}

Eigen::SparseMatrix<double> A = gen_sym_data_sparse(1000);
```

- Then we have to construct a Matrix Product operation, which is provided by Spectra for Sparse Symmetric Eigen matrices. 

`Note: For the solver only a Matrix product operation is required, thus you can specify a custom one without underlying matrix if you wish`

```cpp
#include <Spectra/MatOp/SparseSymMatProd.h>

Spectra::SparseSymMatProd<double> op(mat); // Create the Matrix Product operation
```

- Afterwards the solver can be constructed, and desired parameters can be set 

TODO: explain the constructor (link to doxygen)? Explain that an initial guess can be provided?

```cpp
#include <Spectra/JDSymEigsDPR.h>

Spectra::DavidsonSymEig<OpType> solver(op,2); //Create Solver
```

While their defaults values should be adequate for most situations, several internal parameters of the solver can be tuned :

```cpp

// Maximum size of the search space
solver.setMaxSearchSpaceSize(250); 

// Number of corretion vector to append to the
// search space at each iteration
solver.setCorrectionSize(4);
```

- This solver can then be executed through the compute method, where we also specify which EigenPairs we want through the [Sortrule enum](https://spectralib.org/doc/selectionrule_8h_source). The maximum number of iterations of the solver as well as the convergence criteria for the 
norm of the residues can also be specified in the call of the `compute()` method.

```cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
TEST_CASE("Davidson Symmetric EigenSolver example")
{
    solver.compute(Spectra::SortRule::LargestAlge, maxit = 100, tol=1E-3);
    REQUIRE(solve.info() == CompInfo::Successful);
}
```

- It is also possible to provide a staring values for the eigenvectors. This can be done with the `computeWithGuess` method as illustrated below :

```cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
TEST_CASE("Davidson Symmetric EigenSolver example with guess")
{
    Matrix guess = Eigen::Random(1000, 4);
    Spectra::QR_orthogonalisation(guess);
    solver.computeWithGuess(guess, Spectra::SortRule::LargestAlge, maxit=100, tol=1E-3);
    REQUIRE(solve.info() == CompInfo::Successful);
}
```
