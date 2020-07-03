This is an example of how to use the Jacobi-Davidson Symmetric Eigenvalue Solver with DPR correction method. This test can also be found as a full file in the [test/DavidsonSymEigen_example.ccp](test/DavidsonSymEigen_example.cpp) file and can be compiled with cmake and run afterwards

```bash
mkdir build && cd build && cmake ../
make DavidsonSymEigen_example
./test/DavidsonSymEigen_example
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

- Afterwards the solver can be constructed. Both the operator and the desired number of eigen values must be specfied in the constructor. While their defaults values should be adequate for most situations, several internal parameters of the solver, most notably the maximum size of the search space and the number of correction vectors to append to the search size at each iteration, can be tuned :

```cpp
#include <Spectra/DavidsonSymEig.h>

Spectra::DavidsonSymEig<OpType> solver(op, 2); //Create Solver

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
TEST_CASE("Davidson Symmetric Sparse EigenSolver example with guess")
{
    Eigen::MatrixXd guess = Eigen::MatrixXd::Random(1000, 4);
    Spectra::QR_orthogonalisation(guess);
    solver.computeWithGuess(guess, Spectra::SortRule::LargestAlge, 100, 1E-3);
    REQUIRE(solver.info() == Spectra::CompInfo::Successful);
}
```
- The Davidson solver can also be used in combination with Dense matrices

```cpp
#include <Eigen/Dense>

// Generate data for testing
Eigen::MatrixXd gen_sym_data_dense(int n)
{
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd mat1 = mat + mat.transpose();
    mat1.diagonal().array() = 10* n;
    return mat1;
}

Eigen::MatrixXd B = gen_sym_data_dense(1000);
```

- Create the Matrix Product operator and solver

```cpp
#include <Spectra/MatOp/DenseSymMatProd.h>

Spectra::DenseSymMatProd<double> op_dense(B); // Create the Matrix Product operation

Spectra::DavidsonSymEig<Spectra::DenseSymMatProd<double>> solver_dense(op_dense,2); //Create Solver
```

- Test the dense solver

```cpp
TEST_CASE("Davidson Dense Symmetric EigenSolver example")
{
    Eigen::MatrixXd guess = Eigen::MatrixXd::Random(1000, 4);
    Spectra::QR_orthogonalisation(guess);
    solver_dense.computeWithGuess(guess, Spectra::SortRule::LargestAlge, 100, 1E-3);
    REQUIRE(solver_dense.info() == Spectra::CompInfo::Successful);
}
```