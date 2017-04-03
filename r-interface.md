---
layout: page
title: R Interface
---

# R Interface

### The RSpectra Package

[RSpectra](https://CRAN.R-project.org/package=RSpectra) is the R interface of Spectra.
It provides functions `eigs()` and `eigs_sym()` for eigenvalue problems,
and `svds()` for truncated (partial) SVD. These functions are generic, meaning
that different matrix types in R, including sparse matrices, are supported.

Below is a list of implemented ones:

- `matrix` (defined in base R)
- `dgeMatrix` (defined in **Matrix** package, for general matrices)
- `dsyMatrix` (defined in **Matrix** package, for symmetric matrices)
- `dgCMatrix` (defined in **Matrix** package, for column oriented sparse matrices)
- `dgRMatrix` (defined in **Matrix** package, for row oriented sparse matrices)
- `function` (implicitly specify the matrix by providing a function that calculates matrix product `A %*% x`)

### Quick Examples

<h4><span class="label label-success">Eigenvalue Problems</span></h4>

We first generate some matrices:

<pre><code class="r">library(RSpectra)
library(Matrix)
n = 20
k = 5

set.seed(111)
A1 = matrix(rnorm(n^2), n)  ## class "matrix"
A2 = Matrix(A1)             ## class "dgeMatrix"
</code></pre>

General matrices have complex eigenvalues:

<pre><code class="r">eigs(A1, k)
eigs(A2, k, opts = list(retvec = FALSE))  ## eigenvalues only
</code></pre>

RSpectra also works on sparse matrices:

<pre><code class="r">A1[sample(n^2, n^2 / 2)] = 0
A3 = as(A1, "dgCMatrix")
A4 = as(A1, "dgRMatrix")

eigs(A3, k)
eigs(A4, k)
</code></pre>

Function interface is also supported:

<pre><code class="r">f = function(x, args)
{
    as.numeric(args %*% x)
}
eigs(f, k, n = n, args = A3)
</code></pre>

Symmetric matrices have real eigenvalues.

<pre><code class="r">A5 = crossprod(A1)
eigs_sym(A5, k)
</code></pre>

To find the smallest (in absolute value) `k` eigenvalues of `A5`,
we have two approaches:

<pre><code class="r">eigs_sym(A5, k, which = "SM")
eigs_sym(A5, k, sigma = 0)
</code></pre>

The results should be the same, but the latter method is preferred, since
it is much more stable on large matrices.

<h4><span class="label label-success">SVD Problems</span></h4>

For SVD problems, users can can specify the number of singular values
(`k`), number of left singular vectors (`nu`) and number of right
singular vectors(`nv`).

<pre><code class="r">m = 100
n = 20
k = 5
set.seed(111)
A = matrix(rnorm(m * n), m)

svds(A, k)
svds(t(A), k, nu = 0, nv = 3)
</code></pre>

Similar to `eigs()`, `svds()` supports sparse matrices:

<pre><code class="r">A[sample(m * n, m * n / 2)] = 0
Asp1 = as(A, "dgCMatrix")
Asp2 = as(A, "dgRMatrix")

svds(Asp1, k)
svds(Asp2, k, nu = 0, nv = 0)
</code></pre>

### Reference

The function-by-function reference can be found in
[this manual](https://cran.r-project.org/web/packages/RSpectra/RSpectra.pdf)
and in the built-in help system of R by typing `?RSpectra::eigs` and
`?RSpectra::svds`
