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

include(cmake/FindTorch.cmake)
include(cmake/FindOpenCV.cmake)

add_subdirectory(src)

enable_testing()

add_test(NAME "LoadModel" COMMAND test_loadmodel.exe)
add_test(NAME "ResNet" COMMAND test_resnet.exe ../../../data/image.jpg ../../../model/model/resnet/resnet_model_cpu.pth false)
