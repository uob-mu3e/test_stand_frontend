cmake_minimum_required(VERSION 3.0)

add_executable(crfe crfe.cpp)

target_include_directories(crfe PRIVATE ${CMAKE_SOURCE_DIR}/common/include ${CMAKE_SOURCE_DIR}/common/libmudaq)
target_link_libraries(crfe mfe midas rt dl pthread util clockboard ${Boost_LIBRARIES})



install(TARGETS crfe DESTINATION bin)
