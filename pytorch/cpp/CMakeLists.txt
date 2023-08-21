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
include(cmake/FindCarnot.cmake)
include(cmake/FindCAWFInference.cmake)

find_package(MPI REQUIRED)

add_subdirectory(src)

enable_testing()

add_test(NAME "LoadModel"      COMMAND test_loadmodel.exe --model-file ${CMAKE_HOME_DIRECTORY}/../../model/resnet/resnet_model_cpu.pth)

if(TARGET opencv)
add_test(NAME "ResNet"         COMMAND test_resnet.exe --image-file ${CMAKE_HOME_DIRECTORY}/../../data/my_dog.jpg --model-file ${CMAKE_HOME_DIRECTORY}/../../model/resnet/resnet_model_cpu.pth --labels-file ${CMAKE_HOME_DIRECTORY}/../../data/labels.txt false)
endif()

add_test(NAME "LoadGraph"      COMMAND test_gnn.exe --test-load-graph 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json)
add_test(NAME "GraphToPTGraph" COMMAND test_gnn.exe --test-convert-graph-to-ptgraph 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json)
add_test(NAME "LoadGNNModel"   COMMAND test_gnn.exe --test-load-model 1 --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt)
add_test(NAME "INFERENCE-CPU"  COMMAND test_gnn.exe --test-inference 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt)
add_test(NAME "INFERENCE-GPU"  COMMAND test_gnn.exe --test-inference 1 --graph-file ${CMAKE_HOME_DIRECTORY}/../../data/gnn/graph0.json --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt --use-gpu 1)
add_test(NAME "INF-BATCH-CPU"  COMMAND test_gnn.exe --test-inference 1 --data-dir ${CMAKE_HOME_DIRECTORY}/../../data/gnn --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt --batch-size 4 --nb-graphs 4)
add_test(NAME "INF-BATCH-GPU"  COMMAND test_gnn.exe --test-inference 1 --data-dir ${CMAKE_HOME_DIRECTORY}/../../data/gnn --model-file ${CMAKE_HOME_DIRECTORY}/../../model/gnn/script_best_model_normal.pt --batch-size 4 --nb-graphs 4 --use-gpu 1)

# test without MPI
add_test(NAME "test_LR"        COMMAND test_LR.exe     100 ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/LR_model.pt )
add_test(NAME "test_MLR"       COMMAND test_MLR.exe    100 ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/MLR_model.pt)
add_test(NAME "test_non_LR"    COMMAND test_non_LR.exe 100 ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/non_LR_model.pt)

# test with MPI
if(TARGET MPI::MPI_CXX)
add_test (NAME "test_LR_MPI"      COMMAND mpirun -np 2 ./src/test_LR_MPI.exe 10000  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/LR_model.pt)
add_test (NAME "test_MLR_MPI"     COMMAND mpirun -np 2 ./src/test_MLR_MPI.exe 10000  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/MLR_model.pt)
add_test (NAME "test_non_LR_MPI"  COMMAND mpirun -np 2 ./src/test_non_LR_MPI.exe 10000  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/non_LR_model.pt)
add_test (NAME "benchmark_mpi_pt" COMMAND mpirun -np 2 ./src/test_non_LR_MPI.exe 10000  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/non_LR_model.pt 1 10)
endif()

add_test (NAME "BENCH_LR_pt_cpu" COMMAND benchmark_scale_pt.exe --model-file  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/LR_model.pt --use-gpu 0)
add_test (NAME "BENCH_LR_pt_gpu" COMMAND benchmark_scale_pt.exe --model-file  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/LR_model.pt --use-gpu 1)

add_test (NAME "BENCH_MLR_pt_cpu" COMMAND benchmark_scale_pt.exe --model-file  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/MLR_model.pt --nb-features 2 --use-gpu 0)
add_test (NAME "BENCH_MLR_pt_gpu" COMMAND benchmark_scale_pt.exe --model-file  ${CMAKE_HOME_DIRECTORY}/../../model/simplenets/pytorch/MLR_model.pt --nb-features 2 --use-gpu 1)
