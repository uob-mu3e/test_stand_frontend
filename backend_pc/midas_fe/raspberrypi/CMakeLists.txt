cmake_minimum_required(VERSION 3.0)
set_source_files_properties(rpi_temp.cpp rpi_led.cpp scfe.cpp PROPERTIES LANGUAGE CXX )

add_executable(scfe scfe.cpp)
target_include_directories(scfe PRIVATE ${MIDAS_DIR}/include)
target_link_libraries(scfe.cpp midas)

add_library(scfe STATIC rpi_temp.cpp rpi_led.cpp)
