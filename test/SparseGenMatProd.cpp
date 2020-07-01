
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/Dense>

using namespace Spectra;

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::Index;

template <typename T>
Eigen::SparseMatrix<T> generate_random_sparse(Index rows, Index cols)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    std::vector<Eigen::Triplet<T>> tripletVector;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
            auto v_ij = dist(gen);
            if (v_ij < 0.5)
            {
                //if larger than treshold, insert it
                tripletVector.push_back(Eigen::Triplet<T>(i, j, v_ij));
            }
        }
    Eigen::SparseMatrix<T> mat(rows, cols);
    //create the matrix
    mat.setFromTriplets(tripletVector.begin(), tripletVector.end());

    return mat;
}

template <typename T>
Eigen::SparseMatrix<T> generate_random_sparse_entries(Index rows, Index cols, int entries)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    std::uniform_real_distribution<double> dist_x(0, rows);
    std::uniform_real_distribution<double> dist_y(0, cols);

    std::vector<Eigen::Triplet<T>> tripletVector;
    for (int n = 0; n < entries ; n++) {
        auto v_ij = dist(gen);
        int x_c = std::round(dist_x(gen));
        int y_c = std::round(dist_y(gen));
        //Dont care about duplicates
        tripletVector.push_back(Eigen::Triplet<T>(x_c, y_c, v_ij));
    }
    
    Eigen::SparseMatrix<T> mat(rows, cols);
    //create the matrix
    mat.setFromTriplets(tripletVector.begin(), tripletVector.end());

    return mat;
}

TEMPLATE_TEST_CASE("matrix operations [100x100]", "[SparseGenMatProd]", float, double)
{
    std::srand(123);
    constexpr Index n = 100;

    Eigen::SparseMatrix<TestType> mat1 = generate_random_sparse<TestType>(n, n);
    Eigen::SparseMatrix<TestType> mat2 = generate_random_sparse<TestType>(n, n);

    SparseGenMatProd<TestType> sparse1(mat1);
    Eigen::SparseMatrix<TestType> xs = sparse1 * mat2;
    Eigen::SparseMatrix<TestType> ys = mat1 * mat2;

    INFO("The matrix-matrix product must be the same as in eigen.")
    REQUIRE(xs.isApprox(ys));
    INFO("The accesor operator must produce the same element as in eigen")
    REQUIRE(mat1.coeff(45, 22) == sparse1(45, 22));
}

TEMPLATE_TEST_CASE("matrix operations [100000000x100000000]", "[SparseGenMatProd]", float, double)
{
    std::srand(123);
    constexpr Index n = 100000000; //adding another 0 will explode it, so perhaps it can be more efficient

    Eigen::SparseMatrix<TestType> mat1 = generate_random_sparse_entries<TestType>(n, n, 1000);
    Eigen::SparseMatrix<TestType> mat2 = generate_random_sparse_entries<TestType>(n, n, 1000);

    SparseGenMatProd<TestType> sparse1(mat1);
    INFO("The matrix-matrix product must not explode.")
    Eigen::SparseMatrix<TestType> xs = sparse1 * mat2;
    if (xs.coeff(50000,3042) == 0) {
        xs.coeffRef(50000,3242) = 1; // Dont let eigen optimalize it away
    } else
    {
        xs.coeffRef(50000,3242) = 2; // Dont let eigen optimalize it away
    }
    

    INFO("The accessor operator must not explode")
    REQUIRE(mat1.coeff(45, 22) == sparse1(45, 22));
}
