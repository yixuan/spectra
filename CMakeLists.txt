cmake_minimum_required (VERSION 3.13 FATAL_ERROR)
project (Spectra VERSION 0.8.1 LANGUAGES CXX)

# Supported options
# -----------------
option(BUILD_TESTS "Build tests" OFF)


# Look for supporting libraries
# -----------------------------
find_package(Eigen3 REQUIRED)

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

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION include)

install(EXPORT Spectra-targets
	FILE Spectra-targets.cmake
	NAMESPACE Spectra::
	DESTINATION cmake
	)

# Configure package
# -----------------

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/Spectra-config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/Spectra-config.cmake
    INSTALL_DESTINATION cmake
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/cmake/Spectra-config-version.cmake
    VERSION ${Spectra_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/cmake/Spectra-config.cmake
        ${CMAKE_BINARY_DIR}/cmake/Spectra-config-version.cmake
    DESTINATION cmake
)
