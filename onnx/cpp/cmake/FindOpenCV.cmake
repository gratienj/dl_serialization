#
# Find OpenCV library
#
if(NOT OPENCV_ROOT)
  set(OPENCV_ROOT $ENV{OPENCV_ROOT})
endif()

if(OPENCV_ROOT)
  set(_OPENCV_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_OPENCV_SEARCH_OPTS)
endif()


if(NOT OPENCV_FOUND)
   # pour limiter le mode verbose
  set(OPENCV_FIND_QUIETLY ON)
  find_package(OpenCV COMPONENTS opencv_core opencv_highgui opencv_imgproc opencv_dnn QUIET)
endif()

set(OPENCV_FIND_QUIETLY ON)
find_package_handle_standard_args(OpenCV
        DEFAULT_MSG
        OpenCV_FOUND
        OpenCV_INCLUDE_DIRS
        OpenCV_LIBS)

set(OPENCV_FOUND False)
if(OpenCV_FOUND)
  set(OPENCV_FOUND True)
  set(OPENCV_LIBRARIES ${OpenCV_LIBS})
  set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})

  if(NOT TARGET opencv)

    add_library(opencv UNKNOWN IMPORTED)

    set_target_properties(opencv PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${OPENCV_LIBRARIES}")

    set_target_properties(opencv PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${OPENCV_INCLUDE_DIRS}")
  endif()
endif(OpenCV_FOUND)
