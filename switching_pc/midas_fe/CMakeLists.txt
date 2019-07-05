cmake_minimum_required(VERSION 3.0)

FILE(GLOB_RECURSE INC_ALL "*.h")

add_executable(switch_fe switch_fe.cpp ${MXML_DIR}/mxml ${INC_ALL})
add_executable(FEBrw_slow_fe FEBrw_slow_fe.cpp ${MXML_DIR}/mxml ${MSCB_DIR}/src/mscb ${INC_ALL})

target_include_directories(switch_fe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})
target_link_libraries(switch_fe mfe midas rt pthread util libmudaq)

target_include_directories(FEBrw_slow_fe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MSCB_DIR}/include ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})
target_link_libraries(FEBrw_slow_fe mfe midas rt pthread util)

install(TARGETS switch_fe FEBrw_slow_fe DESTINATION bin)
