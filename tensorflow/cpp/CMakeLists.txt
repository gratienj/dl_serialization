PROJECT(TensorflowCPP)
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

include(cmake/FindTensorFlow.cmake)
include(cmake/FindBoost.cmake)
#FIND_PATH(BOOST_INCLUDE_DIR boost)
#add_library(boost INTERFACE IMPORTED)
#set_target_properties(boost PROPERTIES 
#    INTERFACE_INCLUDE_DIRECTORIES "${BOOST_INCLUDE_DIR}")

find_package(MPI REQUIRED)


add_subdirectory(src)

enable_testing()

add_test(NAME "TestCAPI"    COMMAND test_capi.exe)
add_test(NAME "TestCPPFlow"
         COMMAND test_cppflow.exe ${CMAKE_HOME_DIRECTORY}/../../data/my_cat.jpg ${CMAKE_HOME_DIRECTORY}/../../model/efficientnet/model ${CMAKE_HOME_DIRECTORY}/../../data/imagenet_class_index.json)


# add test LR_model:
add_test(NAME "test_LR" COMMAND test_LR.exe 1000 ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/tensorflow/LR_model.pb)
# add test MLR_model:
add_test(NAME "test_MLR" COMMAND test_MLR.exe 1000 ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/tensorflow/MLR_model.pb)
# add test MLR_model:
add_test(NAME "test_non_LR" COMMAND test_non_LR.exe 1000 ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/tensorflow/non_LR_model.pb)

add_test (NAME "BENCH_TF" COMMAND benchmark_scale_tf.exe ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/tensorflow/LR_model.pb 100 1)
add_test (NAME "BENCH_MPI_TF" COMMAND mpirun -np 2 ./src/benchmark_mpi_tf.exe 64000  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/tensorflow/LR_model.pb 1 10)
