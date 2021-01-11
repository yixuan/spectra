cmake_minimum_required (VERSION 3.5 FATAL_ERROR)
project (Spectra VERSION 0.9.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)  # make CMake look into the ./cmake/ folder for configuration files

option(ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(ENABLE_COVERAGE_BUILD)
    message(STATUS "Enabling coverage build")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()


# Supported options
# -----------------
option(BUILD_TESTS "Build tests" OFF)
include(FeatureSummary) # More verbose Output for libraries using set_package_properties

# Look for supporting libraries
# -----------------------------
find_package(Eigen3 NO_MODULE REQUIRED)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED PURPOSE "C++ vector data structures")
message(STATUS "Found Eigen3 Version: ${Eigen3_VERSION} Path: ${Eigen3_DIR}")

# Setup library
# -------------

add_library(Spectra INTERFACE)
target_include_directories(Spectra
		INTERFACE
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
		)
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)

# Parse additional options
# ------------------------

if(BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()

# Install the library (relative to the CMAKE_INSTALL_PREFIX)
# ----------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS Spectra
    EXPORT Spectra-targets
	INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

install(EXPORT Spectra-targets
	FILE Spectra-targets.cmake
	NAMESPACE Spectra::
	DESTINATION share/spectra/cmake
	)

# Configure package
# -----------------

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/spectra-config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/spectra-config.cmake
    INSTALL_DESTINATION share/spectra/cmake
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/cmake/spectra-config-version.cmake
    VERSION ${Spectra_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/cmake/spectra-config.cmake
        ${CMAKE_BINARY_DIR}/cmake/spectra-config-version.cmake
    DESTINATION share/spectra/cmake
)

find_package(CLANG_FORMAT 7.0.1)
if(CLANG_FORMAT_FOUND)
    file(GLOB_RECURSE FORMAT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()


# Print out summary of all detected packages
feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
