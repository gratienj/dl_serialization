#
# Find OpenCV library
#
if(NOT ONNX_ROOT)
  set(ONNX_ROOT $ENV{ONNX_ROOT})
endif()

if(ONNX_ROOT)
  set(_ONNX_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_ONNX_SEARCH_OPTS)
endif()


message(status " FIND ONNX : ${ONNX_ROOT} ${ONNX_FOUND}")
if(NOT ONNX_FOUND)
   # pour limiter le mode verbose
  set(ONNX_FIND_QUIETLY ON)
  find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime/core/session/onnxruntime_cxx_api.h 
            HINTS ${ONNX_ROOT}/include)

  find_library(ONNX_RUNTIME_LIB onnxruntime 
               HINTS ${ONNX_ROOT}/lib)

  message(status "ONNX INCLUDE DIR : ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
  message(status "ONNX LIB  : ${ONNX_RUNTIME_LIB}")
endif()

set(ONNX_FIND_QUIETLY ON)
find_package_handle_standard_args(ONNX
        DEFAULT_MSG
        ONNX_RUNTIME_SESSION_INCLUDE_DIRS
        ONNX_RUNTIME_LIB)

if(ONNX_FOUND)
  message(status "ONNX FOUND")
  #set(ONNX_LIBRARIES ${ONNX_LIBS})
  #set(ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIRS})

  if(NOT TARGET onnx)
    message(status "ONNX INCLUDE DIR : ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
    add_library(onnx UNKNOWN IMPORTED)

    set_target_properties(onnx PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${ONNX_RUNTIME_LIB}")

    set_target_properties(onnx PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
  else()
    message(status "TARGET ONNX ALLREADY DEFINED")
  endif()
else()
    message(status "ONNX NOT FOUND")
endif(ONNX_FOUND)
