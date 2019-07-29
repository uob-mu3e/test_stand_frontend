cmake_minimum_required(VERSION 3.1)

project(mu3e_online VERSION 0.1)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "" FORCE)
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
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

# on some older boxes, wpedantic and MIDAS do not mix
add_subdirectory(modules)

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

set(MIDAS_DIR ${PROJECT_SOURCE_DIR}/modules/midas)
set(MIDASLIB_DIR ${PROJECT_SOURCE_DIR}/modules/midas/linux/lib)
set(MIDASLIB ${MIDASLIB_DIR}/libmidas.a)
set(MIDASINCLUDE_DIR ${PROJECT_SOURCE_DIR}/modules/midas/include)
set(LIBMUDAQ_DIR ${PROJECT_SOURCE_DIR}/common/libmudaq)
set(LIBMUTRIG_DIR ${PROJECT_SOURCE_DIR}/common/libmutrig)
set(INCMUDAQ_DIR ${PROJECT_SOURCE_DIR}/common/include)
set(MXML_DIR ${PROJECT_SOURCE_DIR}/modules/midas/mxml)
set(MSCB_DIR ${PROJECT_SOURCE_DIR}/modules/midas/mscb)

add_subdirectory(backend_pc)
add_subdirectory(common/libmudaq)
add_subdirectory(common/libmutrig)
add_subdirectory(switching_pc/midas_fe)
add_subdirectory(switching_pc/midas_fe_scifi)
if (CUDA_FOUND)
add_subdirectory(farm_pc/midas_fe)
endif(CUDA_FOUND)


configure_file(set_env.sh.in set_env.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/set_env.sh DESTINATION ./)

configure_file(start_daq.sh.in start_daq.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/start_daq.sh DESTINATION bin
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
