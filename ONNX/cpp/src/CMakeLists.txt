cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(inference.exe inference.cpp)
target_include_directories(inference.exe PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(inference.exe PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(inference.exe boost ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})

INSTALL (TARGETS inference.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")