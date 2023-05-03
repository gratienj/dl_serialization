
# add the executable
add_executable(test_loadmodel.exe test_loadmodel.cpp)
TARGET_LINK_LIBRARIES(test_loadmodel.exe torch boost)

INSTALL (TARGETS test_loadmodel.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

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
endif()

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

