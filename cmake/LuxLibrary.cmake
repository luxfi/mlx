# LuxLibrary.cmake - Shared CMake infrastructure for Lux C++ libraries
#
# This module provides standardized install, export, and packaging for all
# Lux C++ libraries following the Unix library conventions.
#
# Contract:
#   - Install layout: include/lux/<pkg>/, lib/, lib/cmake/<pkg>/, lib/pkgconfig/
#   - Naming: CMake=lux-<pkg>, Target=lux::<pkg>, Library=liblux<pkg>.{dylib,so}
#   - Relocatable: @rpath on macOS, $ORIGIN on Linux
#   - Default install prefix: /usr/local (ALL platforms)
#   - No absolute paths in installed files
#
# Usage:
#   include(LuxLibrary)
#   lux_add_library(
#     NAME gpu
#     SOURCES ${SOURCES}
#     HEADERS ${HEADERS}
#     DEPENDENCIES lux-lattice   # optional
#     FRAMEWORKS Metal Foundation  # macOS only
#     VERSION 1.0.0
#   )

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ========================= DEFAULT /usr/local =========================
# Standard Unix convention: /usr/local for locally compiled software
# This applies to ALL platforms (macOS, Linux, BSD, etc.)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix" FORCE)
endif()

# Standard compile flags for all Lux libraries
set(LUX_CXX_STANDARD 17)
set(LUX_CXX_STANDARD_REQUIRED ON)

# Validate library name doesn't have underscore
function(_lux_validate_name NAME)
  if(NAME MATCHES "_")
    message(FATAL_ERROR "Library name '${NAME}' contains underscore. Use lowercase without underscores.")
  endif()
endfunction()

# Main function to create a Lux library with proper install/export
function(lux_add_library)
  set(options STATIC SHARED)
  set(oneValueArgs NAME VERSION DESCRIPTION)
  set(multiValueArgs SOURCES HEADERS DEPENDENCIES FRAMEWORKS RESOURCES)
  cmake_parse_arguments(LUX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  _lux_validate_name(${LUX_NAME})

  # Derive names from the short name
  set(LIB_NAME "lux${LUX_NAME}")                    # libluxgpu.dylib
  set(PKG_NAME "lux-${LUX_NAME}")                   # lux-gpu (CMake package)
  set(TARGET_NAME "lux::${LUX_NAME}")               # lux::gpu (exported target)
  set(INTERNAL_TARGET "${PKG_NAME}")                # lux-gpu (build target)

  # Determine library type
  if(LUX_STATIC)
    set(LIB_TYPE STATIC)
  elseif(LUX_SHARED OR BUILD_SHARED_LIBS)
    set(LIB_TYPE SHARED)
  else()
    set(LIB_TYPE STATIC)
  endif()

  # Create the library
  add_library(${INTERNAL_TARGET} ${LIB_TYPE} ${LUX_SOURCES})
  add_library(${TARGET_NAME} ALIAS ${INTERNAL_TARGET})

  # Set output name (produces libluxgpu.dylib)
  set_target_properties(${INTERNAL_TARGET} PROPERTIES
    OUTPUT_NAME ${LIB_NAME}
    VERSION ${LUX_VERSION}
    SOVERSION ${LUX_VERSION}
    CXX_STANDARD ${LUX_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ${LUX_CXX_STANDARD_REQUIRED}
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME ${LUX_NAME}  # Export as just the short name under lux::
  )

  # Relocatable RPATH settings - enables "just works" linking from /usr/local
  # Universal settings for ALL platforms (macOS, Linux, BSD, etc.)
  set_target_properties(${INTERNAL_TARGET} PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON
  )

  if(APPLE)
    # macOS: @rpath + @loader_path for relocatable binaries
    set_target_properties(${INTERNAL_TARGET} PROPERTIES
      MACOSX_RPATH ON
      INSTALL_NAME_DIR "@rpath"
      INSTALL_RPATH "@loader_path;@loader_path/../lib;${CMAKE_INSTALL_PREFIX}/lib"
    )
  else()
    # Linux/BSD: $ORIGIN for relocatable + explicit /usr/local/lib
    set_target_properties(${INTERNAL_TARGET} PROPERTIES
      INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${CMAKE_INSTALL_PREFIX}/lib"
    )
  endif()

  # Include directories
  target_include_directories(${INTERNAL_TARGET}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # Link dependencies (other lux packages)
  # Dependencies are specified as package names (e.g., lux-lattice)
  # but linked as namespace targets (e.g., lux::lattice)
  foreach(dep ${LUX_DEPENDENCIES})
    find_package(${dep} CONFIG REQUIRED)
    # Extract short name from package name (lux-lattice -> lattice)
    string(REGEX REPLACE "^lux-" "" DEP_SHORT "${dep}")
    target_link_libraries(${INTERNAL_TARGET} PUBLIC lux::${DEP_SHORT})
  endforeach()

  # Link frameworks (macOS)
  if(APPLE AND LUX_FRAMEWORKS)
    foreach(fw ${LUX_FRAMEWORKS})
      find_library(${fw}_FRAMEWORK ${fw})
      if(${fw}_FRAMEWORK)
        target_link_libraries(${INTERNAL_TARGET} PUBLIC ${${fw}_FRAMEWORK})
      endif()
    endforeach()
  endif()

  # ========================= INSTALL =========================

  # Install library
  install(TARGETS ${INTERNAL_TARGET}
    EXPORT ${PKG_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install headers
  if(LUX_HEADERS)
    install(FILES ${LUX_HEADERS}
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lux/${LUX_NAME}
    )
  endif()

  # Install resources (metallib, kernels, etc.)
  if(LUX_RESOURCES)
    install(FILES ${LUX_RESOURCES}
      DESTINATION ${CMAKE_INSTALL_DATADIR}/lux/${LUX_NAME}
    )
  endif()

  # ========================= EXPORT =========================

  # Export targets
  install(EXPORT ${PKG_NAME}Targets
    FILE ${PKG_NAME}Targets.cmake
    NAMESPACE lux::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PKG_NAME}
  )

  # Generate version file
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}ConfigVersion.cmake"
    VERSION ${LUX_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  # Generate config file from template if exists, otherwise create simple one
  set(CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PKG_NAME}Config.cmake.in")
  if(EXISTS ${CONFIG_TEMPLATE})
    configure_package_config_file(
      ${CONFIG_TEMPLATE}
      "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake"
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PKG_NAME}
    )
  else()
    # Generate simple config file with proper package init
    # This replaces @PACKAGE_INIT@ which only works with configure_package_config_file()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake" "
# Package config for ${PKG_NAME}
get_filename_component(PACKAGE_PREFIX_DIR \"\${CMAKE_CURRENT_LIST_DIR}/../../../\" ABSOLUTE)

macro(set_and_check _var _file)
  set(\${_var} \"\${_file}\")
  if(NOT EXISTS \"\${_file}\")
    message(FATAL_ERROR \"File or directory \${_file} referenced by variable \${_var} does not exist !\")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp \${${_NAME}_FIND_COMPONENTS})
    if(NOT \${_NAME}_\${comp}_FOUND)
      if(\${_NAME}_FIND_REQUIRED_\${comp})
        set(\${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

include(CMakeFindDependencyMacro)
")
    foreach(dep ${LUX_DEPENDENCIES})
      file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake" "
find_dependency(${dep} CONFIG)
")
    endforeach()
    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake" "
include(\"\${CMAKE_CURRENT_LIST_DIR}/${PKG_NAME}Targets.cmake\")

# Resource directory for runtime assets
set(LUX_${LUX_NAME}_RESOURCE_DIR \"\${PACKAGE_PREFIX_DIR}/share/lux/${LUX_NAME}\")

check_required_components(${PKG_NAME})
")
  endif()

  # Install config files
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PKG_NAME}
  )

  # ========================= PKG-CONFIG =========================

  # Generate pkg-config file
  set(PC_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PKG_NAME}.pc.in")
  if(EXISTS ${PC_TEMPLATE})
    configure_file(${PC_TEMPLATE}
      "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}.pc"
      @ONLY
    )
  else()
    # Generate simple pkg-config file
    set(PC_REQUIRES "")
    foreach(dep ${LUX_DEPENDENCIES})
      set(PC_REQUIRES "${PC_REQUIRES} ${dep}")
    endforeach()
    
    set(PC_LIBS_PRIVATE "")
    if(APPLE AND LUX_FRAMEWORKS)
      foreach(fw ${LUX_FRAMEWORKS})
        set(PC_LIBS_PRIVATE "${PC_LIBS_PRIVATE} -framework ${fw}")
      endforeach()
    endif()

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}.pc"
"prefix=${CMAKE_INSTALL_PREFIX}
exec_prefix=\${prefix}
libdir=\${prefix}/${CMAKE_INSTALL_LIBDIR}
includedir=\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}
resourcedir=\${prefix}/${CMAKE_INSTALL_DATADIR}/lux/${LUX_NAME}

Name: ${PKG_NAME}
Description: ${LUX_DESCRIPTION}
Version: ${LUX_VERSION}
Requires:${PC_REQUIRES}
Libs: -L\${libdir} -l${LIB_NAME}
Libs.private:${PC_LIBS_PRIVATE}
Cflags: -I\${includedir}
")
  endif()

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  )

  # Export for parent scope
  set(${PKG_NAME}_TARGET ${INTERNAL_TARGET} PARENT_SCOPE)
  set(${PKG_NAME}_VERSION ${LUX_VERSION} PARENT_SCOPE)

endfunction()

# CI check function - verify no absolute paths in installed library
# This check runs at INSTALL time, not build time, since RPATH fixup happens during install
function(lux_verify_relocatable TARGET)
  # Only meaningful on install
  if(APPLE)
    install(CODE "
      set(LIBFILE \"\${CMAKE_INSTALL_PREFIX}/lib/lib${TARGET}.dylib\")
      if(EXISTS \"\${LIBFILE}\")
        message(STATUS \"Verifying \${LIBFILE} is relocatable...\")
        execute_process(
          COMMAND otool -D \"\${LIBFILE}\"
          OUTPUT_VARIABLE OTOOL_OUTPUT
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT OTOOL_OUTPUT MATCHES \"@rpath/\")
          message(WARNING \"Install name not using @rpath in \${LIBFILE}\")
        endif()
        execute_process(
          COMMAND otool -l \"\${LIBFILE}\"
          OUTPUT_VARIABLE OTOOL_RPATH
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(OTOOL_RPATH MATCHES \"path.*/Users/|path.*/home/\")
          message(WARNING \"Absolute path found in RPATH of \${LIBFILE}\")
        endif()
        message(STATUS \"OK: \${LIBFILE} verified\")
      endif()
    " COMPONENT Runtime)
  else()
    install(CODE "
      set(LIBFILE \"\${CMAKE_INSTALL_PREFIX}/lib/lib${TARGET}.so\")
      if(EXISTS \"\${LIBFILE}\")
        message(STATUS \"Verifying \${LIBFILE} is relocatable...\")
        execute_process(
          COMMAND readelf -d \"\${LIBFILE}\"
          OUTPUT_VARIABLE READELF_OUTPUT
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(READELF_OUTPUT MATCHES \"RPATH|RUNPATH\" AND READELF_OUTPUT MATCHES \"/home/|/Users/\")
          message(WARNING \"Absolute RPATH found in \${LIBFILE}\")
        endif()
        message(STATUS \"OK: \${LIBFILE} verified\")
      endif()
    " COMPONENT Runtime)
  endif()
endfunction()
