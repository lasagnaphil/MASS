cmake_minimum_required(VERSION 2.8.6)
project(mss)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

option(MASS_BUILD_RENDERER "Build renderer for MASS" ON)

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)

add_compile_options(-DHAVE_CSTDDEF)

add_definitions(-DMASS_ROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")
set(CMAKE_CXX_STANDARD 14)

add_subdirectory( core )
add_subdirectory( python )
if (MASS_BUILD_RENDERER)
    add_subdirectory( render )
endif()
