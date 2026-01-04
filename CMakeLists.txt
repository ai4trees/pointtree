cmake_minimum_required(VERSION 3.30...3.35)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
include(ExternalProject)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if(APPLE)
    execute_process(COMMAND brew --prefix OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${BREW_PREFIX}/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")
    include_directories("${BREW_PREFIX}/opt/libomp/include")
    link_directories("${BREW_PREFIX}/opt/libomp/lib")
endif()

find_package(OpenMP REQUIRED)

set(EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip")
set(EIGEN_SOURCE_DIR "${CMAKE_BINARY_DIR}/eigen3")
set(EIGEN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/eigen3/eigen-3.4.0")
set(EIGEN_ZIP_PATH "${EIGEN_SOURCE_DIR}/eigen-3.4.0.zip")
set(EIGEN_CORE_HEADER_PATH "${EIGEN_INCLUDE_DIR}/Eigen/Core")

if (NOT EXISTS ${EIGEN_CORE_HEADER_PATH})
  message(STATUS "Downloading Eigen.")
  file(MAKE_DIRECTORY ${EIGEN_SOURCE_DIR})
  file(DOWNLOAD ${EIGEN_URL} ${EIGEN_ZIP_PATH} SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${EIGEN_ZIP_PATH} DESTINATION ${EIGEN_SOURCE_DIR})
else()
    message(STATUS "Found Eigen source directory. Skipping download.")
endif()
include_directories( ${EIGEN_INCLUDE_DIR} )

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

set(NANOFLANN_URL "https://codeload.github.com/jlblancoc/nanoflann/zip/refs/tags/v1.6.2")
set(NANOFLANN_SOURCE_DIR "${CMAKE_BINARY_DIR}/nanoflann")
set(NANOFLANN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/nanoflann/nanoflann-1.6.2/include")
set(NANOFLANN_ZIP_PATH "${NANOFLANN_SOURCE_DIR}/nanoflann-v1.6.2.zip")
set(NANONFLANN_HEADER_PATH "${NANOFLANN_INCLUDE_DIR}/nanoflann.hpp")

if (NOT EXISTS ${NANONFLANN_HEADER_PATH})
  message(STATUS "Downloading nanoflann.")
  file(MAKE_DIRECTORY ${NANOFLANN_SOURCE_DIR})
  file(DOWNLOAD ${NANOFLANN_URL} ${NANOFLANN_ZIP_PATH} SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${NANOFLANN_ZIP_PATH} DESTINATION ${NANOFLANN_SOURCE_DIR})
else()
    message(STATUS "Found nanoflann source directory. Skipping download.")
endif()

include_directories( ${NANOFLANN_INCLUDE_DIR} )

pybind11_add_module(_tree_x_algorithm_cpp MODULE
                    csrc/pybind/instance_segmentation/tree_x_algorithm_pybind.cpp)
target_link_libraries(_tree_x_algorithm_cpp PRIVATE OpenMP::OpenMP_CXX)

pybind11_add_module(_coarse_to_fine_algorithm_cpp MODULE
                    csrc/pybind/instance_segmentation/coarse_to_fine_algorithm_pybind.cpp)
target_link_libraries(_coarse_to_fine_algorithm_cpp PRIVATE OpenMP::OpenMP_CXX)

pybind11_add_module(_evaluation_cpp MODULE
                    csrc/pybind/evaluation_pybind.cpp)
target_link_libraries(_evaluation_cpp PRIVATE OpenMP::OpenMP_CXX)

pybind11_add_module(_operations_cpp MODULE
                    csrc/pybind/operations_pybind.cpp)
target_link_libraries(_operations_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS _coarse_to_fine_algorithm_cpp DESTINATION "${SKBUILD_PROJECT_NAME}")
install(TARGETS _tree_x_algorithm_cpp DESTINATION "${SKBUILD_PROJECT_NAME}")
install(TARGETS _evaluation_cpp DESTINATION "${SKBUILD_PROJECT_NAME}")
install(TARGETS _operations_cpp DESTINATION "${SKBUILD_PROJECT_NAME}")
