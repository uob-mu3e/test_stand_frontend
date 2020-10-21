cmake_minimum_required(VERSION 3.0)

add_executable(switch_fe_mupix switch_fe.cpp ${INC_ALL})
#KB: disable most annoying compiler warnings that will not go away
target_compile_options(switch_fe_mupix PRIVATE 
	-Wno-missing-field-initializers
	-Wno-unused-parameter
)
message("${INCMUTRIG_DIR}")
target_include_directories(switch_fe_mupix PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR} ${LIBMUPIX_DIR}/include)
target_link_libraries(switch_fe_mupix mfe midas dl rt pthread util libmudaq_scifi libmutrig libmupix dl)


install(TARGETS switch_fe_mupix DESTINATION bin)
