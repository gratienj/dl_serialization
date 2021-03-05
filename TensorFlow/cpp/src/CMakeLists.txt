
# add the executable
add_executable(test_capi.exe test_capi.cpp)
TARGET_LINK_LIBRARIES(test_capi.exe tensorflow)

add_executable(test_cppflow.exe test_cppflow.cpp)
TARGET_LINK_LIBRARIES(test_cppflow.exe tensorflow boost)


INSTALL (TARGETS test_capi.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
INSTALL (TARGETS test_cppflow.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
