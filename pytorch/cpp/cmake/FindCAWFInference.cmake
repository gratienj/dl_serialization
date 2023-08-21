#
# Find CAWFInference includes
#
# This module defines
# CAWFINFERENCE_INCLUDE_DIRS, where to find headers,
# CAWFINFERENCE_LIBRARIES, the libraries to link against to use CAWFInference.
# CAWFINFERENCE_FOUND If false, do not try to use CAWFInference.



if(NOT CAWFINFERENCE_ROOT)
  set(CAWFINFERENCE_ROOT $ENV{CAWFINFERENCE_ROOT})
endif()

if(CAWFINFERENCE_ROOT)
  set(_CAWFINFERENCE_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_CAWFINFERENCE_SEARCH_OPTS)
endif()

#CAWFInference
find_library(CAWFINFERENCE_LIBRARY
             NAMES CAWFInference
             HINTS ${CAWFINFERENCE_ROOT} 
             PATH_SUFFIXES lib
            )
mark_as_advanced(CAWFINFERENCE_LIBRARY)
	
find_path(CAWFINFERENCE_INCLUDE_DIR cawf_inference/cawf_api.h
  HINTS ${CAWFINFERENCE_ROOT} 
  PATH_SUFFIXES include
  ${_CAWFINFERENCE_SEARCH_OPTS}
  )
mark_as_advanced(CAWFINFERENCE_INCLUDE_DIR)

find_library(INFERENCE_GRPC_PROTO_LIBRARY
             NAMES inference_grpc_proto
             HINTS ${CAWFINFERENCE_ROOT} 
             PATH_SUFFIXES lib
            )
mark_as_advanced(INFERENCE_GRPC_PROTO_LIBRARY)

find_package_handle_standard_args(CAWFINFERENCE 
	DEFAULT_MSG 
	CAWFINFERENCE_INCLUDE_DIR 
	CAWFINFERENCE_LIBRARY
	INFERENCE_GRPC_PROTO_LIBRARY)

if(CAWFINFERENCE_FOUND AND NOT TARGET cawfinference)

  # Find Protobuf installation
  # Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
  set(protobuf_MODULE_COMPATIBLE TRUE)
  #find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${Protobuf_VERSION}")

  set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
  set(_REFLECTION gRPC::grpc++_reflection)
  
  # Find gRPC installation
  # Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
  #find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")

  set(_GRPC_GRPCPP gRPC::grpc++)
  
  add_library(inference_grpc_proto UNKNOWN IMPORTED)
  set_target_properties(inference_grpc_proto PROPERTIES
                     IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                     IMPORTED_LOCATION "${INFERENCE_GRPC_PROTO_LIBRARY}")
                     
  #Construction de cawfinference
  add_library(cawfinference UNKNOWN IMPORTED)
  set_target_properties(cawfinference PROPERTIES
                     IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                     IMPORTED_LOCATION "${CAWFINFERENCE_LIBRARY}")
                     
  set_property(TARGET cawfinference 
               APPEND PROPERTY INTERFACE_LINK_LIBRARIES  
                        inference_grpc_proto
                        #${_REFLECTION}
                        #${_GRPC_GRPCPP}
                       ${_PROTOBUF_LIBPROTOBUF})
  
  set_target_properties(cawfinference PROPERTIES 
	INTERFACE_INCLUDE_DIRECTORIES "${CAWFINFERENCE_INCLUDE_DIR}")
  add_definitions(-DUSE_CAWFINFERENCE)
  #target_compile_definitions(cawfinference PRIVATE $<TARGET_PROPERTY:target_compile_definitions,INTERFACE_COMPILE_DEFINITIONS> -DUSE_CAWFINFERENCE)
endif()
