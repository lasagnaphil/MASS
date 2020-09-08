cmake_minimum_required(VERSION 2.8.6)
project(pymss)

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

# include(FindOpenMP)
# if(OPENMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

find_package(DART REQUIRED COMPONENTS collision-bullet CONFIG)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(TBB)

link_directories(../core/)
include_directories(../core/)

file(GLOB srcs "*.h" "*.cpp")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${DART_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
add_definitions(${TBB_DEFINITIONS})

add_library(pymss SHARED ${srcs})
if (TBB_FOUND)
    target_compile_definitions(pymss PUBLIC -DCOMPILE_MULTI_ENV_MANAGER)
    message(STATUS "TBB Found. Will compile MultiEnvManager")
else()
    message(STATUS "TBB Not Found. Will not compile MultiEnvManager")
endif(TBB_FOUND)

target_link_libraries(pymss ${DART_LIBRARIES} ${PYTHON_LIBRARIES} ${TBB_LIBRARIES} mss pybind11::module)
set_target_properties(pymss PROPERTIES PREFIX "" )
