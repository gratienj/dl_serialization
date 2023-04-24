
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
