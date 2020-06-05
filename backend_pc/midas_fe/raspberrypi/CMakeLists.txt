cmake_minimum_required(VERSION 3.1)

project(mu3e_pi VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# include midas
add_subdirectory(../../../modules midas)
set(MIDAS_DIR ${PROJECT_SOURCE_DIR}/../../../modules/midas)

set_source_files_properties(ucfe.cpp PROPERTIES LANGUAGE CXX )
add_executable(usb_clock_pi ucfe.cpp)
target_include_directories(usb_clock_pi PRIVATE ${MIDAS_DIR}/include)
target_link_libraries(usb_clock_pi midas)

