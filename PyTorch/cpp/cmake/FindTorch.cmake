#
# Find Torch library
#
if(NOT TORCH_ROOT)
  set(TORCH_ROOT $ENV{TORCH_ROOT})
endif()

if(TORCH_ROOT)
  set(_TORCH_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_TORCH_SEARCH_OPTS)
endif()


if(NOT TORCH_FOUND)
   # pour limiter le mode verbose
  set(TORCH_FIND_QUIETLY ON)
  find_package(Torch)
endif()

if(NOT TORCH_FOUND)

  find_library(TORCH_LIBRARY
               NAMES torch
               HINTS ${TORCH_ROOT}
               PATH_SUFFIXES lib lib64
               ${_TORCH_SEARCH_OPTS}
    )

  mark_as_advanced(TORCH_LIBRARY)


  find_path(TORCH_INCLUDE_DIR torch/script.h
            HINTS ${TORCH_ROOT}
            PATH_SUFFIXES include
            ${_TORCH_SEARCH_OPTS}
    )
  mark_as_advanced(TORCH_INCLUDE_DIR)

  find_path(TORCH_CAPI_INCLUDE_DIR torch/torch.h
            HINTS ${TORCH_ROOT}
            PATH_SUFFIXES include/torch/csrc/api/include
            ${_TORCH_SEARCH_OPTS}
    )
  mark_as_advanced(TORCH_CAPI_INCLUDE_DIR)

endif()
# pour limiter le mode verbose
set(TORCH_FIND_QUIETLY ON)


find_package_handle_standard_args(TORCH
        DEFAULT_MSG
        TORCH_INCLUDE_DIR
        TORCH_LIBRARY)

if(TORCH_FOUND)
  if(NOT TARGET torch)

    add_library(torch UNKNOWN IMPORTED)

    set_target_properties(torch PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${TORCH_LIBRARIES}")

    #set_target_properties(torch PROPERTIES
    #      INTERFACE_INCLUDE_DIRECTORIES "${TORCH_CAPI_INCLUDE_DIR}")
    set_target_properties(torch PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")

  endif()

endif()
