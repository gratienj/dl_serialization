cmake_minimum_required(VERSION 3.13)

project(TensorRT_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(tensorrt_cpp_api SHARED engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(inference.exe inference.cpp)
TARGET_LINK_LIBRARIES(inference.exe boost tensorrt_cpp_api ${OpenCV_LIB_COMPONENTS})

INSTALL (TARGETS inference.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                           benchmark_scale_tensorrt
#/////////////////////////////////////////////////////////////////////////////////
add_library(trt_engine SHARED TRTEngine.cpp utils/logger.cpp)
target_include_directories(trt_engine PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(trt_engine PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(benchmark_scale_tensorrt.exe benchmark_scale_tensorrt.cpp)
target_link_libraries(benchmark_scale_tensorrt.exe boost trt_engine TensorRT::TensorRT)

#install:
INSTALL (TARGETS benchmark_scale_tensorrt.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                             benchmark_mpi_onnx :        MPI
#/////////////////////////////////////////////////////////////////////////////////

#add_executable(benchmark_mpi_onnx.exe benchmark_mpi_onnx.cpp)
#target_link_libraries(benchmark_mpi_onnx.exe boost onnx MPI::MPI_CXX)

#install:
#INSTALL (TARGETS benchmark_mpi_onnx.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")