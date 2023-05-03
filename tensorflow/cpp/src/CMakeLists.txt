
# add the executable
add_executable(test_capi.exe test_capi.cpp)
TARGET_LINK_LIBRARIES(test_capi.exe tensorflow)

add_executable(test_cppflow.exe test_cppflow.cpp)
TARGET_LINK_LIBRARIES(test_cppflow.exe tensorflow boost)

add_executable(test_carnot.exe test_carnot.cpp cnpy.cpp)
TARGET_LINK_LIBRARIES(test_carnot.exe tensorflow boost z)

INSTALL (TARGETS test_capi.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS test_cppflow.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS test_carnot.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

# openMP:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# add executable LR_model :-------------------------------------
add_executable(test_LR.exe test_LR.cpp)

# target link libraries:
target_link_libraries(test_LR.exe tensorflow tensorflow_framework)
# set property:
set_property(TARGET test_LR.exe PROPERTY CXX_STANDARD 17)

# add executable MLR_model :-------------------------------------
add_executable(test_MLR.exe test_MLR.cpp)
# target link libraries:
target_link_libraries(test_MLR.exe tensorflow tensorflow_framework)
# set property:
set_property(TARGET test_MLR.exe PROPERTY CXX_STANDARD 17)

# add executable non_LR_model :-------------------------------------
add_executable(test_non_LR.exe test_non_LR.cpp)

# target link libraries:
target_link_libraries(test_non_LR.exe tensorflow tensorflow_framework)
# set property:
set_property(TARGET test_non_LR.exe PROPERTY CXX_STANDARD 17)


# add executable non_LR_model :-------------------------------------
add_executable(benchmark_scale_tf.exe benchmark_scale_tf.cpp)

# target link libraries:
target_link_libraries(benchmark_scale_tf.exe boost tensorflow tensorflow_framework)
# set property:
set_property(TARGET benchmark_scale_tf.exe PROPERTY CXX_STANDARD 17)

add_executable(benchmark_mpi_tf.exe benchmark_mpi_tf.cpp)
#add target libraries:
target_link_libraries(benchmark_mpi_tf.exe boost tensorflow tensorflow_framework MPI::MPI_CXX)

# set property:
set_property(TARGET benchmark_mpi_tf.exe PROPERTY CXX_STANDARD 17)


#install:
INSTALL (TARGETS test_LR.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS test_MLR.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS test_non_LR.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS benchmark_scale_tf.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS benchmark_mpi_tf.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
