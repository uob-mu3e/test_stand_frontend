cmake_minimum_required(VERSION 3.0)

FILE(GLOB_RECURSE INC_ALL "*.h")

add_executable(malibu_fe malibu_fe.cpp ${MXML_DIR}/mxml ${INC_ALL})

target_include_directories(malibu_fe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})
target_link_libraries(malibu_fe mfe midas dl rt pthread util libmudaq libmalibu) #TODO: Here is the place to add STiC lib or malibu lib

install(TARGETS malibu_fe DESTINATION bin)

#==========================make file for scifi ===============================
#add_executable(switch_fe_scifi switch_fe.cpp ${MXML_DIR}/mxml ${INC_ALL})
#KB: disable most annoying compiler warnings that will not go away
#target_compile_options(switch_fe_scifi PRIVATE
	#        -Wno-missing-field-initializers
	#        -Wno-unused-parameter
	#)
#target_include_directories(switch_fe_scifi PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR})
#target_link_libraries(switch_fe_scifi mfe midas rt pthread util libmudaq libmutrig)


#install(TARGETS switch_fe_scifi DESTINATION bin)
#==============================================================================
