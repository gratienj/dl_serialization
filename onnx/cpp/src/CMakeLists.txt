cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(inference.exe inference.cpp)
TARGET_LINK_LIBRARIES(inference.exe boost onnx ${OpenCV_LIB_COMPONENTS})

INSTALL (TARGETS inference.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                           benchmark_scale_onnx  numactl
#/////////////////////////////////////////////////////////////////////////////////

add_executable(benchmark_scale_onnx.exe benchmark_scale_onnx.cpp)
target_link_libraries(benchmark_scale_onnx.exe boost onnx)

#install:
INSTALL (TARGETS benchmark_scale_onnx.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                             benchmark_mpi_onnx :        MPI
#/////////////////////////////////////////////////////////////////////////////////

add_executable(benchmark_mpi_onnx.exe benchmark_mpi_onnx.cpp)
target_link_libraries(benchmark_mpi_onnx.exe boost onnx MPI::MPI_CXX)

#install:
INSTALL (TARGETS benchmark_mpi_onnx.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")