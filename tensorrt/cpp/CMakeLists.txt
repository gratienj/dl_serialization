PROJECT(TorchCPP)
cmake_minimum_required(VERSION 3.16)


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
ENDIF (NOT CMAKE_BUILD_TYPE)

if(NOT CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install/${PROJECT_VERSION}
      CACHE PATH "Default install path" FORCE)
endif()

include(CMakeParseArguments)
include(FindPackageHandleStandardArgs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/FindBoost.cmake)
include(cmake/FindOpenCV.cmake)
#include(cmake/FindONNX.cmake)
include(cmake/FindTensorRT.cmake)

find_package(CUDA REQUIRED)

#include(cmake/FindTorch.cmake)
#find_package(OpenCV REQUIRED)

# find MPI------------------------------------------
find_package(MPI REQUIRED)


add_subdirectory(src)

enable_testing()

add_test(NAME "INFERENCE" COMMAND inference.exe --data-dir ${CMAKE_HOME_DIRECTORY}/../../data/onnx)
#add_test(NAME "BENCH-SCL" COMMAND benchmark_scale_onnx.exe --model-file ../../../model/simplenets/onnx/C2DNN1.onnx --nb-samples 100 --nb-features 1 --nb-calls 10)
#add_test(NAME "BENCH-MPI" COMMAND mpirun -np 2 ./src/benchmark_mpi_onnx.exe --model-file ../../../model/simplenets/onnx/C2DNN1.onnx --nb-samples 100 --nb-features 1 --nb-calls 1 10)
