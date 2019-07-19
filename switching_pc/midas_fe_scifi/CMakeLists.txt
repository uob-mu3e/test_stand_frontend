cmake_minimum_required(VERSION 3.0)

add_executable(switch_fe_scifi switch_fe.cpp ${MXML_DIR}/mxml ${INC_ALL})

target_include_directories(switch_fe_scifi PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR})
target_link_libraries(switch_fe_scifi mfe midas rt pthread util libmudaq libmutrig)


install(TARGETS switch_fe_scifi DESTINATION bin)
