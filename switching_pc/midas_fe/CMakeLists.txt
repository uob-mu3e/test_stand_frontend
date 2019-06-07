cmake_minimum_required(VERSION 3.0)

add_executable(switch_fe switch_fe.cpp ${MXML_DIR}/mxml)

target_include_directories(switch_fe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR})
target_link_libraries(switch_fe mfe midas rt pthread util)


install(TARGETS switch_fe DESTINATION bin)
