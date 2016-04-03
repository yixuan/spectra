## [Unreleased]
### Added
- Added the wrapper classes `SparseSymMatProd` and `SparseSymShiftSolve`
  for sparse symmetric matrices
- Added the wrapper class `SparseGenRealShiftSolve` for general sparse matrices
- Added tests for sparse matrices
- Using Travis CI for automatic unit test

### Changed
- Updated included [Catch](https://github.com/philsquared/Catch) to v1.4.0



## [0.2.0] - 2016-02-28
### Added
- Benchmark script now outputs number of matrix operations
- Added this change log
- Added a simple built-in random number generator, so that the algorithm
  was made to be deterministic
- Added the wrapper class `DenseSymMatProd` for symmetric matrices

### Changed
- Improved Arnoldi factorization
  - Iteratively corrects orthogonality
  - Creates new residual vector when invariant subspace is found
  - Stability for matrices with repeated eigenvalues is greatly improved
- Adjusted deflation tolerance in double shift QR
- Updated result analyzer
- Updated included [Catch](https://github.com/philsquared/Catch) to v1.3.4
- Updated copyright information
- **API change**: Default operator of `SymEigsSolver` was changed from
  `DenseGenMatProd` to `DenseSymMatProd`



## [0.1.0] - 2015-12-19
### Added
- Initial release of Spectra
