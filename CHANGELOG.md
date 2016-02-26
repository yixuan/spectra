## [Unreleased]
### Added
- Benchmark script now outputs number of matrix operations
- Added this change log

### Changed
- Improved Arnoldi factorization
  - Iteratively corrects orthogonality
  - Creates new residual vector when invariant subspace is found
  - Stability for matrices with repeated eigenvalues is greatly improved
- Adjusted deflation tolerance in double shift QR
- Updated result analyzer
- Updated included Catch to v1.3.4

## [0.1.0] - 2015-12-19
### Added
- Initial release of Spectra
