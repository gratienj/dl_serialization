#
# Find TensorFlow library
#
if(NOT TENSORFLOW_ROOT)
  set(TENSORFLOW_ROOT $ENV{TENSORFLOW_ROOT})
endif()

if(TENSORFLOW_ROOT)
  set(_TENSORFLOW_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_TENSORFLOW_SEARCH_OPTS)
endif()


if(NOT TENSORFLOW_FOUND)
  # pour limiter le mode verbose
  set(TENSORFLOW_FIND_QUIETLY ON)
  #find_package(Tensorflow)
endif()

if(NOT TENSORFLOW_FOUND)

  find_library(TENSORFLOW_LIBRARY
               NAMES tensorflow 
               HINTS ${TENSORFLOW_ROOT}
               PATH_SUFFIXES lib
               ${_TENSORFLOW_SEARCH_OPTS}
    )

  mark_as_advanced(TENSORFLOW_LIBRARY)

  find_path(TENSORFLOW_INCLUDE_DIR tensorflow/c/c_api.h
            HINTS ${TENSORFLOW_ROOT} 
            PATH_SUFFIXES include
            ${_TENSORFLOW_SEARCH_OPTS}
    )
  mark_as_advanced(TENSORFLOW_INCLUDE_DIR)

endif()

# pour limiter le mode verbose
set(TENSORFLOW_FIND_QUIETLY ON)

find_package_handle_standard_args(TENSORFLOW
        DEFAULT_MSG
        TENSORFLOW_INCLUDE_DIR
        TENSORFLOW_LIBRARY)


if(TENSORFLOW_FOUND)
  

  if(NOT TARGET tensorflow)

    add_library(tensorflow UNKNOWN IMPORTED)

    set_target_properties(tensorflow PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${TENSORFLOW_LIBRARY}")

    set_target_properties(tensorflow PROPERTIES 
          INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_INCLUDE_DIR}")


  endif()

endif()
