cmake_minimum_required(VERSION 2.8.11)
# suppress policy warnings on newer cmake versions
if(POLICY CMP0043)
  cmake_policy(SET CMP0043 NEW)
endif()

project(mu3edaq)

# set default built-type to "Release"
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
# set default install directory to "install" inside the repository
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "install prefix" FORCE)
endif()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
# always add the current (binary) directory to the list of include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# show which Geant4 installation is used
set(Geant4_CONFIG_DEBUG TRUE)

# required packages
find_package(Boost 1.49 REQUIRED COMPONENTS thread system program_options)
find_package(Git 1.8 REQUIRED)
#find_package(ROOT 5.34 REQUIRED)
# optional packages
find_package(Doxygen 1.8)
if(APPLE)
    find_package(OpenGl)
else()
    find_package(OpenGL)
endif()

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

# set common include directories
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
include_directories(SYSTEM ${ROOT_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/libmudaq)

#midas and related packages work with classical makefile
add_custom_target(
        midas
        ALL
        COMMAND make
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/modules/midas)

    set(MIDASDIR ${CMAKE_CURRENT_LIST_DIR}/modules/midas)
    set(MIDASLIBDIR ${CMAKE_CURRENT_LIST_DIR}/modules/midas/linux/lib)
    set(MIDASLIB ${MIDASLIBDIR}/libmidas.a)
    set(MIDASINCLUDEDIR ${CMAKE_CURRENT_LIST_DIR}/modules/midas/include)


add_custom_target(
        mscb
        ALL
        COMMAND make
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/modules/mscb)

add_custom_target(
        exptab
        ALL
        COMMAND echo "mu3e . $ENV{USER}" > ${CMAKE_CURRENT_LIST_DIR}/mhttpd/exptab
)

add_subdirectory(common/libmudaq)

if (CUDA_FOUND)
add_subdirectory(farm_pc/midas_fe)
endif(CUDA_FOUND)

# install activate script
configure_file(activate.sh.in activate.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/activate.sh DESTINATION ./)
