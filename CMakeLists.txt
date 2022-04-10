cmake_minimum_required (VERSION 3.5 FATAL_ERROR)
project (Spectra VERSION 1.0.1 LANGUAGES CXX)

# Make CMake look into the ./cmake/ folder for configuration files
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(FeatureSummary)
include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Without compiler extensions like gnu++11
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g") # Use slight optimization to speed up tests
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Supported options
# -----------------
option(ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
include(FeatureSummary) # More verbose Output for libraries using set_package_properties

# Look for supporting libraries
# -----------------------------
find_package(Eigen3 NO_MODULE REQUIRED HI)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED PURPOSE "C++ vector data structures")
message(STATUS "Found Eigen3 Version: ${Eigen3_VERSION} Path: ${Eigen3_DIR}")

# Setup library
# -------------
add_library(Spectra INTERFACE)
target_include_directories(
    Spectra
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)
add_library(Spectra::Spectra ALIAS Spectra)

# Parse additional options
# ------------------------
if(ENABLE_COVERAGE_BUILD)
    message(STATUS "Enabling coverage build")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Install the library (relative to the CMAKE_INSTALL_PREFIX)
# ----------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(INSTALL_CMAKE_DIR share/spectra/cmake
  CACHE PATH "Relative path for installing cmake config files.")

install(
  TARGETS Spectra
  EXPORT Spectra-Targets
  INCLUDES DESTINATION include)

# Install the include directory
install(
  DIRECTORY
  ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include)

# Install targets to the install-tree
install(
  EXPORT Spectra-Targets
  FILE spectra-targets.cmake
  NAMESPACE Spectra::
  DESTINATION ${INSTALL_CMAKE_DIR})

# Add all targets to the build-tree export set
export(
  TARGETS Spectra
  FILE ${PROJECT_BINARY_DIR}/spectra-targets.cmake)

# Export the package for use from the build-tree (this registers the build-tree
# with the CMake user package registry.)
export(PACKAGE Spectra)

# This is the build-tree version
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/spectra-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/spectra-config.cmake
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# This is the version that will be installed
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/spectra-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/spectra-config.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKE_DIR})

# Write the version file (same for build and install tree)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/spectra-config-version.cmake
  VERSION ${Spectra_VERSION}
  COMPATIBILITY AnyNewerVersion)

# Install the config files
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/spectra-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/spectra-config-version.cmake
  DESTINATION ${INSTALL_CMAKE_DIR})

find_package(CLANG_FORMAT 9)
if(CLANG_FORMAT_FOUND)
    file(GLOB_RECURSE FORMAT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
        DEPENDS ${FORMAT_SOURCES}
    )
endif()

# Print out summary of all detected packages
feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)

