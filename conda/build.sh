#!/usr/bin/env bash

mkdir build && cd build
if [ `uname` == Darwin ]; then
    ${BUILD_PREFIX}/bin/cmake \
        .. \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CLANG} \
        -DCMAKE_C_COMPILER=${CLANGXX} \
        -DCMAKE_C_FLAGS="${CFLAGS} ${OPTS}" \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS} ${OPTS}" \
        -DCMAKE_VERBOSE_MAKEFILE=TRUE \
        -DBUILD_TESTS=TRUE
fi
if [ `uname` == Linux ]; then
    ${BUILD_PREFIX}/bin/cmake \
        .. \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${GCC} \
        -DCMAKE_C_COMPILER=${GXX} \
        -DCMAKE_C_FLAGS="${CFLAGS} ${OPTS}" \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS} ${OPTS}" \
        -DCMAKE_VERBOSE_MAKEFILE=TRUE \
        -DBUILD_TESTS=TRUE
fi

make -j${CPU_COUNT}
make test ARGS=-j${CPU_COUNT}
make install

