cmake_minimum_required(VERSION 3.10)
project(flashprog)

if (NOT CMAKE_CXX_STANDARD)    # let top-level project overwrite c++ standard
   set(CMAKE_CXX_STANDARD 17)
endif()

add_executable(flashprog ${MIDASSYS}/mscb/src/mscb.cxx ${MIDASSYS}/mscb/src/mxml.cxx ${MIDASSYS}/mscb/src/strlcpy.cxx main.cpp)

target_include_directories(flashprog PRIVATE ${MIDASSYS}/mscb/include)
target_link_libraries(flashprog PRIVATE pthread)

install(TARGETS flashprog DESTINATION bin)
