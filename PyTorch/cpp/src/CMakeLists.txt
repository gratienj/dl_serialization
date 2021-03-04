
# add the executable
add_executable(test_loadmodel.exe test_loadmodel.cpp)
TARGET_LINK_LIBRARIES(test_loadmodel.exe torch)

INSTALL (TARGETS test_loadmodel.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


# add the executable
add_executable(test_resnet.exe test_resnet.cpp opencvutils.cpp torchutils.cpp)
TARGET_LINK_LIBRARIES(test_resnet.exe torch ${OpenCV_LIB_COMPONENTS})

INSTALL (TARGETS test_resnet.exe RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
