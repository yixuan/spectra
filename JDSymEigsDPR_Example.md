Hello World

This is an example of how to use the Jacobi-Davidson Symmetric Eigenvalue Solver with DPR correction method. This test can also be found as a full file in the [test/JDSymEigsDPR_example.ccp](test/JDSymEigsDPR_example.cpp) file and can be compiled with cmake and run afterwards

```bash
mkdir build && cd build && cmake ../
make JDSymEigsDPR_example
./test/JDSymEigsDPR_example
```

Suppose we want to find the 2 eigenpairs with the Largest value from a 1000x1000 Sparse Matrix A, then we could use this solver to quickly find them.


- First we have to construct the matrix
```cpp
#include <Eigen/Sparse>

// Generate Matrix
Eigen::SparseMatrix<double> gen_sym_data_sparse(int n)
{
    double prob = 0.5;
    SpMatrix<T> mat(n, n);
    std::default_random_engine gen;
    gen.seed(0);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (distr(gen) < prob)
                mat.insert(i, j) = distr(gen) - 0.5;
        }
    }
    return mat;
}

Eigen::SparseMatrix<double> A = gen_sym_data_sparse(1000)
```

- Then we have to construct a Matrix Product operation, which is provided by Spectra for Sparse Eigen matrices. 

`Note: For the solver only a Matrix product operation is required, thus you can specify a custom one without underlying matrix if you wish`

