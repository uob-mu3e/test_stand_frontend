#
# CMakeLists.txt for Mu3e Environment Frontend
#
# SR 12 April 2022
#

cmake_minimum_required(VERSION 3.0)
project(env_fe)

# set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Check for environment variables
if (NOT DEFINED ENV{MIDASSYS})
   message(SEND_ERROR "MIDASSYS environment variable not defined.")
endif()
set(MIDASSYS $ENV{MIDASSYS})

# enable certain compile warnings
add_compile_options(-Wall)

# ass required linux libraries
if (${CMAKE_SYSTEM_NAME} MATCHES Linux)
   list(APPEND LIBS -lpthread -lutil -lrt -ldl)
endif()

# main frontend executable
add_executable(env_fe
   src/env_fe.cxx
   ${MIDASSYS}/src/mfe.cxx
   ${MIDASSYS}/src/mfed.cxx
   ${MIDASSYS}/drivers/class/multi.cxx
   ${MIDASSYS}/drivers/device/mscbdev.cxx
   ${MIDASSYS}/mscb/src/mscb.cxx)

target_include_directories(env_fe PRIVATE
   ${MIDASSYS}/include 
   ${MIDASSYS}/drivers
   ${MIDASSYS}/mxml
   ${MIDASSYS}/mscb/include)

target_link_libraries(env_fe ${MIDASSYS}/lib/libmidas.a ${LIBS})
