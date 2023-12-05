
set(DSSSolver_VERSION "1.0.0")

add_library(DSSSolver SHARED DSSSolver.h DSSSolver.cpp graph.h graphutils.cpp modelutils.cpp cnpy.cpp)
TARGET_LINK_LIBRARIES(DSSSolver torch torch_sparse torch_scatter boost  z dl)

INSTALL (TARGETS DSSSolver 
         EXPORT DSSSolverTargets
         RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
         INCLUDES DESTINATION include)

install(DIRECTORY .
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ml4cfd"
        FILES_MATCHING PATTERN "*.h"
       )

install(EXPORT DSSSolverTargets
    FILE DSSSolverTargets.cmake
    NAMESPACE ml4cfd::
    DESTINATION lib/cmake/ml4cfd)

include(CMakePackageConfigHelpers)


configure_package_config_file(DSSSolverConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DSSSolverConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml4cfd)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/DSSSolverConfigVersion.cmake
  VERSION ${DSSSolver_VERSION}
  COMPATIBILITY SameMajorVersion )

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DSSSolverConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/DSSSolverConfigVersion.cmake"
        DESTINATION lib/cmake/ml4cfd)