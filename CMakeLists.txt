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

add_subdirectory(modules)
add_subdirectory(backend_pc)

configure_file(set_env.sh.in set_env.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/set_env.sh DESTINATION ./)
