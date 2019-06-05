cmake_minimum_required(VERSION 3.1)

project(mu3e_online VERSION 0.1)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "" FORCE)
endif()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# MU3E_ONLINE_GIT_REF
#find_package(Git 1.8 REQUIRED)
#execute_process(
#    COMMAND ${GIT_EXECUTABLE} describe --abbrev=16 --dirty
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#    OUTPUT_VARIABLE MU3E_ONLINE_GIT_REF
#)
#message(STATUS "MU3E_ONLINE_GIT_REF: ${MU3E_ONLINE_GIT_REF}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    "-Wpedantic" "-Wall" "-Wextra"
    "-Wformat=2"
)

# required packages
find_package(Boost 1.49 REQUIRED COMPONENTS thread system program_options)
find_package(Git 1.8 REQUIRED)
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
find_package(ROOT 5.34 REQUIRED)
# optional packages
find_package(Doxygen 1.8)
message("here we set GLVND to default (https://cmake.org/cmake/help/git-stage/policy/CMP0072.html)")
set(OpenGL_GL_PREFERENCE GLVND)
if(APPLE)
    find_package(OpenGl)
else()
    find_package(OpenGL)
endif()

find_package(CUDA)
find_package(CUDA)

if (CUDA_FOUND)
    message("-- CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} was found at ${CUDA_TOOLKIT_ROOT_DIR}")
    list(APPEND CUDA_NVCC_FLAGS "-arch=compute_30;-DVERBOSE;-g")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
else(CUDA_FOUND)
    message("* CUDA is not found")
    #   message(FATAL_ERROR "Not all CUDA libraries are found")
    message("Not all CUDA libraries are found")
endif(CUDA_FOUND)

add_subdirectory(modules)
set(MIDASDIR ${CMAKE_CURRENT_LIST_DIR}/modules/midas)
set(MIDASLIBDIR ${CMAKE_CURRENT_LIST_DIR}/modules/midas/linux/lib)
set(MIDASLIB ${MIDASLIBDIR}/libmidas.a)
set(MIDASINCLUDEDIR ${CMAKE_CURRENT_LIST_DIR}/modules/midas/include)

add_subdirectory(backend_pc)
add_subdirectory(common/libmudaq)
set(LIBMUDAQDIR ${CMAKE_CURRENT_LIST_DIR}/common/libmudaq)

if (CUDA_FOUND)
add_subdirectory(farm_pc/midas_fe)
endif(CUDA_FOUND)

configure_file(set_env.sh.in set_env.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/set_env.sh DESTINATION ./)
