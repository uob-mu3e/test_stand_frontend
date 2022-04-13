#
# CMakeLists.txt for Mu3e High Voltage Frontend
#
# SR 13 April 2022
#

cmake_minimum_required(VERSION 3.0)
project(hv_fe)

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
add_executable(hv_fe
   src/hv_fe.cxx
   ${MIDASSYS}/src/mfe.cxx
   ${MIDASSYS}/src/mfed.cxx
   ${MIDASSYS}/drivers/class/hv.cxx
   ${MIDASSYS}/drivers/device/mscbhv4.cxx
   ${MIDASSYS}/mscb/src/mscb.cxx)

target_include_directories(hv_fe PRIVATE
   ${MIDASSYS}/include 
   ${MIDASSYS}/drivers
   ${MIDASSYS}/mxml
   ${MIDASSYS}/mscb/include)

target_link_libraries(hv_fe ${MIDASSYS}/lib/libmidas.a ${LIBS})
