cmake_minimum_required(VERSION 3.0)

add_executable(ana ana.cpp)

target_include_directories(ana PRIVATE
    ${MIDAS_DIR}/include
    ${ROOTANA_INCLUDE_DIRS}
    ${ROOT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(ana midas rt pthread ${ROOTANASYS})



