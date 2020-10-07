cmake_minimum_required(VERSION 3.0)

add_executable(switch_fe switch_fe.cpp ${MXML_DIR}/mxml ${INC_ALL})
#KB: disable most annoying compiler warnings that will not go away
target_compile_options(switch_fe PRIVATE 
	-Wno-missing-field-initializers
	-Wno-unused-parameter
)
target_include_directories(switch_fe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})
target_link_libraries(switch_fe mfe midas rt pthread util libmudaq_scifi libmutrig libmupix dl)

install(TARGETS switch_fe DESTINATION bin)
