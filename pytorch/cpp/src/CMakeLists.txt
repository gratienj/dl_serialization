
# add the executable
add_executable(test_loadmodel.exe test_loadmodel.cpp)
TARGET_LINK_LIBRARIES(test_loadmodel.exe torch boost)
INSTALL (TARGETS test_loadmodel.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

if(TARGET TensorRT::TensorRT)
add_library(trt_engine SHARED tensorrt/TRTEngine.cpp tensorrt/TRTEngineTester.cpp utils/9.2/logger.cpp)
target_include_directories(trt_engine PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_include_directories(trt_engine PRIVATE ${TorchCPP_SOURCE_DIR}/src)
target_link_libraries(trt_engine PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
endif()

add_executable(test_ptcarnot.exe test_carnot.cpp cnpy.cpp carnot/PTFlash.cpp)
target_include_directories(test_ptcarnot.exe PRIVATE ${TorchCPP_SOURCE_DIR}/src)
TARGET_LINK_LIBRARIES(test_ptcarnot.exe boost z dl)
if(TARGET carnot)
TARGET_LINK_LIBRARIES(test_ptcarnot.exe carnot)
endif()
if(TARGET cawfinference)
TARGET_LINK_LIBRARIES(test_ptcarnot.exe cawfinference)
endif()
if(TARGET torch)
TARGET_LINK_LIBRARIES(test_ptcarnot.exe torch)
endif()
if(TARGET onnx)
TARGET_LINK_LIBRARIES(test_ptcarnot.exe onnx)
endif()
if(TARGET TensorRT::TensorRT)
TARGET_LINK_LIBRARIES(test_ptcarnot.exe trt_engine)
endif()
INSTALL (TARGETS test_ptcarnot.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

if(TARGET opencv)
# add the executable
add_executable(test_resnet.exe test_resnet.cpp opencvutils.cpp torchutils.cpp)
TARGET_LINK_LIBRARIES(test_resnet.exe boost torch ${OpenCV_LIB_COMPONENTS})

INSTALL (TARGETS test_resnet.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

if(TARGET torch_sparse)
# add the executable
add_executable(test_gnn.exe test_gnn.cpp graphutils.cpp modelutils.cpp)
TARGET_LINK_LIBRARIES(test_gnn.exe torch torch_sparse torch_scatter boost)

INSTALL (TARGETS test_gnn.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

add_subdirectory(ml4cfd)

add_executable(test_dss.exe test_dss.cpp cnpy.cpp ml4cfd/graphutils.cpp ml4cfd/modelutils.cpp)

TARGET_LINK_LIBRARIES(test_dss.exe DSSSolver boost  z dl)
target_include_directories(test_dss.exe PRIVATE ${TorchCPP_SOURCE_DIR}/src)
if(TARGET torch)
TARGET_LINK_LIBRARIES(test_dss.exe torch torch_sparse torch_scatter)
endif()
if(TARGET onnx)
TARGET_LINK_LIBRARIES(test_dss.exe onnx)
endif()
if(TARGET TensorRT::TensorRT)
TARGET_LINK_LIBRARIES(test_dss.exe trt_engine TensorRT::TensorRT)
endif()

INSTALL (TARGETS test_dss.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()


if(TARGET cawfinference)

add_executable(test_cawf.exe test_cawf.cpp)
target_compile_features(test_cawf.exe PRIVATE cxx_std_20)
target_include_directories(test_cawf.exe PRIVATE ${Boost_INCLUDE_DIR})

target_link_libraries(test_cawf.exe cawfinference carnot ${Boost_LIBRARIES} dl z)
#    ${_REFLECTION}
#    ${_GRPC_GRPCPP}
#    ${_PROTOBUF_LIBPROTOBUF})

INSTALL (TARGETS test_cawf.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


add_executable(test_geoxim.exe test_geoxim.cpp)
target_compile_features(test_geoxim.exe PRIVATE cxx_std_20)
target_include_directories(test_geoxim.exe PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(test_geoxim.exe carnot cawfinference carnot ${Boost_LIBRARIES} dl z)
#    ${_REFLECTION}
#    ${_GRPC_GRPCPP}
#    ${_PROTOBUF_LIBPROTOBUF})

INSTALL (TARGETS test_geoxim.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

endif(cawfinference)



add_executable(test_trt.exe test_trt.cpp)
target_include_directories(test_trt.exe PRIVATE ${TorchCPP_SOURCE_DIR}/src)
if(TARGET TensorRT::TensorRT)
TARGET_LINK_LIBRARIES(test_trt.exe trt_engine TensorRT::TensorRT)
endif()

INSTALL (TARGETS test_dss.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


find_package( Threads REQUIRED )

#/////////////////////////////////////////////////////////////////////////////////
#                                     test LR
#/////////////////////////////////////////////////////////////////////////////////
# test LR: --------------------------------------------------------------------
add_executable(test_LR.exe test_LR.cpp )

#link target libraries
target_link_libraries(test_LR.exe torch)
#install:
INSTALL (TARGETS test_LR.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

# test LR MPI: -------------------------------------------------------------------
add_executable(test_LR_MPI.exe test_LR_MPI.cpp)
#add target libraries:
target_link_libraries(test_LR_MPI.exe torch MPI::MPI_CXX)
#install:
INSTALL (TARGETS test_LR_MPI.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


#/////////////////////////////////////////////////////////////////////////////////
#                                     test MLR
#/////////////////////////////////////////////////////////////////////////////////
# test MLR: -------------------------------------------------------------------
add_executable(test_MLR.exe test_MLR.cpp)

#add target libraries:
target_link_libraries(test_MLR.exe torch)
#install:
INSTALL (TARGETS test_MLR.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

# test MLR MPI: -------------------------------------------------------------------
add_executable(test_MLR_MPI.exe test_MLR_MPI.cpp)

#add target libraries:
target_link_libraries(test_MLR_MPI.exe torch MPI::MPI_CXX)
#install:
INSTALL (TARGETS test_MLR_MPI.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                                     test nonLR
#/////////////////////////////////////////////////////////////////////////////////
# test non-LR: -------------------------------------------------------------------
add_executable(test_non_LR.exe test_non_LR.cpp)

#add target libraries:
target_link_libraries(test_non_LR.exe torch)
#install:
INSTALL (TARGETS test_non_LR.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

# test non-LR MPI: -------------------------------------------------------------------
add_executable(test_non_LR_MPI.exe test_non_LR_MPI.cpp)

#add target libraries:
target_link_libraries(test_non_LR_MPI.exe torch MPI::MPI_CXX)
#install:
INSTALL (TARGETS test_non_LR_MPI.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                                     benchmark_scale
#/////////////////////////////////////////////////////////////////////////////////
# test non-LR: -------------------------------------------------------------------
add_executable(benchmark_scale_pt.exe bench/benchmark_scale_pt.cpp)
#add target libraries:
target_link_libraries(benchmark_scale_pt.exe torch boost)
#install:
INSTALL (TARGETS benchmark_scale_pt.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

#/////////////////////////////////////////////////////////////////////////////////
#                                     MPI all
#/////////////////////////////////////////////////////////////////////////////////
# test LR MPI: -------------------------------------------------------------------
add_executable(benchmark_mpi_pt.exe bench/benchmark_mpi_pt.cpp)
#add target libraries:
target_link_libraries(benchmark_mpi_pt.exe torch MPI::MPI_CXX boost)
#target_link_libraries(benchmark_mpi_pt.exe mpi)



#install:
INSTALL (TARGETS benchmark_mpi_pt.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

