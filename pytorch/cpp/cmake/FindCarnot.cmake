#
# Find the CARNOT includes and library
#
# This module uses
# CARNOT_ROOT
#
# This module defines
# CARNOT_FOUND
# CARNOT_INCLUDE_DIRS
# CARNOT_LIBRARIES
#
# Target carnot
#
# Remarque
# pas tres propre Pourrait etre spliter en 2 finders
## un  pour Carnot (le finder Carnot devrait etre mis a disposition par l'equipe Carnot)
## un autre pour Carnot Interface

if(NOT CARNOT_ROOT)
  set(CARNOT_ROOT $ENV{CARNOT_ROOT})
endif()
if($ENV{CARNOT_ECPA})
  set(USE_ECPA TRUE)
  add_definitions(-DCARNOT_ECPA=1)
  set(CARNOT_INCLUDE_PATH ${CARNOT_ROOT}/include/carnot_interface_c)
else()
  set(USE_ECPA FALSE)
  set(CARNOT_INCLUDE_PATH ${CARNOT_ROOT}/include)
endif()
if(WIN32)
  set(CARNOT_LIBRARY_PATH ${CARNOT_ROOT}/lib/Windows10.0.19042/64bits/MSVC_19.29.30146.0/Release)
else()
  #set(CARNOT_LIBRARY_PATH ${CARNOT_ROOT}/lib/Linux3.10.0-1127.el7.x86_64/64bits/GNU_7.3.0/Release
  set(CARNOT_LIBRARY_PATH ${CARNOT_ROOT}/lib/Linux4.18.0-425.19.2.el8_7.x86_64/64bits/GNU_11.2.0/Release)
endif()

if(CARNOT_ROOT)
  set(_CARNOT_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_CARNOT_SEARCH_OPTS)
endif()

if(NOT CARNOT_FOUND) 
 
  # Carnot
  if(NOT WIN32)
      find_library(CARNOT_LIBRARY 
            NAMES carnot
            PATHS ${CARNOT_LIBRARY_PATH} 
            ${_CARNOT_SEARCH_OPTS})
      mark_as_advanced(CARNOT_LIBRARY)
  endif()
  # find_library(HUBOPT_CARNOT_LIBRARY
  #      NAMES ALLHubOptDLL
  #      PATHS ${CARNOT_LIBRARY_PATH} 
  #      ${_CARNOT_SEARCH_OPTS})
  # mark_as_advanced(HUBOPT_CARNOT_LIBRARY)
  if(NOT WIN32)
      find_library(XERCES_CARNOT_LIBRARY
        NAMES xerces-c-3.1
        PATHS ${CARNOT_LIBRARY_PATH} 
        ${_CARNOT_SEARCH_OPTS})
      mark_as_advanced(XERCES_CARNOT_LIBRARY)
  endif()
  # Carnot Interface
  if(NOT WIN32)
    find_library(CARNOT_INTERFACE_LIBRARY carnot_interface
      PATHS ${CARNOT_LIBRARY_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CARNOT_INTERFACE_LIBRARY)
  endif()
  
  if(USE_ECPA)
      find_library(CARNOT_INTERFACE_C_LIBRARY carnot_interface_c
        PATHS ${CARNOT_LIBRARY_PATH} NO_DEFAULT_PATH)
        mark_as_advanced(CARNOT_INTERFACE_C_LIBRARY)
      find_path(CARNOT_INCLUDE_DIR Interface.h cInterface.h
        PATHS ${CARNOT_INCLUDE_PATH} ${CARNOT_INCLUDE_PATH}/carnot_interface ${CARNOT_INCLUDE_PATH}/carnot_interface_c
        ${_CARNOT_SEARCH_OPTS}
      )
  else()
      find_path(CARNOT_INCLUDE_DIR Interface.h
        PATHS ${CARNOT_INCLUDE_PATH} ${CARNOT_INCLUDE_PATH}/carnot_interface
        ${_CARNOT_SEARCH_OPTS}
      )
  endif()
  mark_as_advanced(CARNOT_INCLUDE_DIR)
endif()

if(USE_ECPA)
    if(NOT WIN32)
      find_package_handle_standard_args(Carnot 
        DEFAULT_MSG 
        CARNOT_INCLUDE_DIR
        CARNOT_INTERFACE_LIBRARY
        CARNOT_INTERFACE_C_LIBRARY
        CARNOT_LIBRARY
        XERCES_CARNOT_LIBRARY
        )
    else()
      find_package_handle_standard_args(Carnot
        DEFAULT_MSG 
        CARNOT_INCLUDE_DIR
        CARNOT_INTERFACE_C_LIBRARY
        )
      find_package_handle_standard_args(CarnotECPA
        DEFAULT_MSG 
        CARNOT_INCLUDE_DIR
        CARNOT_INTERFACE_C_LIBRARY
        )
    endif()
else()
  find_package_handle_standard_args(Carnot 
    DEFAULT_MSG 
    CARNOT_INCLUDE_DIR
    CARNOT_INTERFACE_LIBRARY
    CARNOT_LIBRARY
    XERCES_CARNOT_LIBRARY
    )
endif()

# pour limiter le mode verbose
set(CARNOT_FIND_QUIETLY ON)
if(CARNOT_FOUND AND NOT TARGET carnot)
  if(NOT WIN32)
      set(CARNOT_INCLUDE_DIRS ${CARNOT_INCLUDE_DIR} ${CARNOT_INCLUDE_DIR}/Eigen_3.3.7)
      
      set(CARNOT_LIBRARIES 
            ${CARNOT_INTERFACE_LIBRARY}
            ${CARNOT_LIBRARY} 
            ${XERCES_CARNOT_LIBRARY})
            
      
      # Dependances Carnot
      
      ## Xerces
      add_library(xerces_carnot UNKNOWN IMPORTED)
      
      set_target_properties(xerces_carnot PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${XERCES_CARNOT_LIBRARY}")
        
      ## Hubopt
      
      # add_library(hubopt_carnot UNKNOWN IMPORTED)
      
      # set_target_properties(hubopt_carnot PROPERTIES
      #  IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      #  IMPORTED_LOCATION "${HUBOPT_CARNOT_LIBRARY}")
      ## to do ajouter dependance de hubopt vers gfort
      
      # Carnot

      add_library(carnot_main UNKNOWN IMPORTED)
      
      set_target_properties(carnot_main PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
          IMPORTED_LOCATION "${CARNOT_LIBRARY}")
      
      set_property(TARGET carnot_main APPEND PROPERTY 
          INTERFACE_LINK_LIBRARIES "xerces_carnot")
          
      add_definitions(-DUSE_CARNOT)
        
      # set_property(TARGET carnot_main APPEND PROPERTY 
      #  INTERFACE_LINK_LIBRARIES "hubopt_carnot")
        
      # Carnot Interface
      if(NOT USE_ECPA})
        add_library(carnot SHARED IMPORTED)
      
        set_target_properties(carnot PROPERTIES 
          INTERFACE_INCLUDE_DIRECTORIES "${CARNOT_INCLUDE_DIRS}") 
        
        set_target_properties(carnot PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
          IMPORTED_LOCATION "${CARNOT_INTERFACE_LIBRARY}")
      
        set_property(TARGET carnot APPEND PROPERTY 
          INTERFACE_LINK_LIBRARIES "carnot_main")
      else()
        
        # Carnot Interface
        add_library(carnotecpa SHARED IMPORTED)
      
        set_target_properties(carnotecpa PROPERTIES 
          INTERFACE_INCLUDE_DIRECTORIES "${CARNOT_INCLUDE_DIRS}") 
        
        set_target_properties(carnotecpa PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
          IMPORTED_LOCATION "${CARNOT_INTERFACE_LIBRARY}")

        set_target_properties(carnotecpa PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES "C"
          IMPORTED_LOCATION "${CARNOT_INTERFACE_C_LIBRARY}")
      
        set_property(TARGET carnotecpa APPEND PROPERTY 
          INTERFACE_LINK_LIBRARIES "carnot_main")
      endif()
    else()
      if(USE_ECPA)
        set(CARNOT_INCLUDE_DIRS ${CARNOT_INCLUDE_DIR})
        set(CARNOT_LIBRARIES ${CARNOT_INTERFACE_C_LIBRARY})
        # Carnot Interface
        add_library(carnotcinterface UNKNOWN IMPORTED)
      
        set_target_properties(carnotcinterface PROPERTIES 
          INTERFACE_INCLUDE_DIRECTORIES "${CARNOT_INCLUDE_DIRS}") 

        set_target_properties(carnotcinterface PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES "C"
          IMPORTED_LOCATION "${CARNOT_INTERFACE_C_LIBRARY}")
          
        add_library(carnotecpa INTERFACE IMPORTED)
        set_property(TARGET carnotecpa APPEND PROPERTY 
        INTERFACE_LINK_LIBRARIES "carnotcinterface")
        
        file(COPY ${CARNOT_LIBRARY_PATH}/carnot.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
        file(COPY ${CARNOT_LIBRARY_PATH}/xerces-c_3_1.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
        file(COPY ${CARNOT_LIBRARY_PATH}/carnot_interface_c.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
        file(COPY ${CARNOT_LIBRARY_PATH}/carnot_interface.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
        file(COPY ${CARNOT_LIBRARY_PATH}/ALLHubOptDLL.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

      endif()
    endif()
endif()
