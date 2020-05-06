cmake_minimum_required (VERSION 3.10 FATAL_ERROR)
project (Spectra VERSION 0.8.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)  # make CMake look into the ./cmake/ folder for configuration files

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
	DESTINATION cmake
	)

# Configure package
# -----------------

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/spectra-config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/spectra-config.cmake
    INSTALL_DESTINATION cmake
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
    DESTINATION cmake
)



# Print out summary of all detected packages
feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
