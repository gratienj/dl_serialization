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
include(cmake/FindTorch.cmake)
include(cmake/FindOpenCV.cmake)

add_subdirectory(src)

enable_testing()

add_test(NAME "LoadModel"      COMMAND test_loadmodel.exe --model-file ${CMAKE_HOME_DIRECTORY}/../../model/resnet/resnet_model_cpu.pth)
add_test(NAME "ResNet"         COMMAND test_resnet.exe --image-file ${CMAKE_HOME_DIRECTORY}/../../data/my_dog.jpg --model-file ${CMAKE_HOME_DIRECTORY}/../../model/resnet/resnet_model_cpu.pth --labels-file ${CMAKE_HOME_DIRECTORY}/../../data/labels.txt false)

add_test(NAME "LoadGraph"      COMMAND test_gnn.exe --test-load-graph 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json)
add_test(NAME "GraphToPTGraph" COMMAND test_gnn.exe --test-convert-graph-to-ptgraph 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json)
add_test(NAME "LoadGNNModel"   COMMAND test_gnn.exe --test-load-model 1 --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt)
add_test(NAME "INFERENCE-CPU"  COMMAND test_gnn.exe --test-inference 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt)
add_test(NAME "INFERENCE-GPU"  COMMAND test_gnn.exe --test-inference 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt --use-gpu 1)
add_test(NAME "INF-BATCH-CPU"  COMMAND test_gnn.exe --test-inference 1 --data-dir ${CMAKE_HOME_DIRECTORY}/../../data/gnn --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt --batch-size 4 --nb-graphs 4)
add_test(NAME "INF-BATCH-GPU"  COMMAND test_gnn.exe --test-inference 1 --data-dir ${CMAKE_HOME_DIRECTORY}/../../data/gnn --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt --batch-size 4 --nb-graphs 4 --use-gpu 1)
