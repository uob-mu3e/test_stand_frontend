cmake_minimum_required(VERSION 3.16)
project(flashprog)

set(CMAKE_CXX_STANDARD 17)

add_executable(flashprog ../../../modules/midas/mscb/src/mscb.cxx ../../../modules/midas/mscb/src/mxml.cxx ../../../modules/midas/mscb/src/strlcpy.cxx main.cpp)

target_include_directories(flashprog PRIVATE ../../../modules/midas/mscb/include)

install(TARGETS flashprog DESTINATION bin)
