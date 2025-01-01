set(test_target_sources)

add_library (tests-main tests-main.cpp)
list(APPEND test_target_sources
        # Linear algebra
        QR.cpp
        Eigen.cpp
        Schur.cpp
        BKLDLT.cpp
        Arnoldi.cpp

        # JD linear algebra
        Orthogonalization.cpp
        RitzPairs.cpp
        SearchSpace.cpp

        # Matrix operators
        DenseGenMatProd.cpp
        DenseSymMatProd.cpp
        SparseGenMatProd.cpp
        SparseSymMatProd.cpp

        # Symmetric eigen solver
        SymEigs.cpp
        SymEigsShift.cpp

        # Hermitian eigen solver
        HermEigs.cpp

        # General eigen solver
        GenEigs.cpp
        GenEigsRealShift.cpp
        GenEigsComplexShift.cpp

        # Symmetric generalized eigen solver
        SymGEigsCholesky.cpp
        SymGEigsRegInv.cpp
        SymGEigsShift.cpp

        # SVD
        SVD.cpp

        # JD eigen solver
        JDSymEigsBase.cpp
        JDSymEigsDPRConstructor.cpp
        DavidsonSymEigs.cpp

        # Examples from bug reports
        Example1.cpp
        Example2.cpp
        Example3.cpp
        )

foreach(TEST_SOURCE ${test_target_sources})
    # Extract the filename without extension (NAME_WE) as a name for our executable
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Add an executable based on the source
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # Configure (include headers and link libraries) the test
    target_link_libraries(${TEST_NAME} PRIVATE Spectra tests-main)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})  # the working directory is the out-of-source build directory
endforeach()
