cmake_minimum_required(VERSION 3.0)

add_executable(switch_fe switch_fe.cpp)
#KB: disable most annoying compiler warnings that will not go away
target_compile_options(switch_fe PRIVATE
	-Wno-missing-field-initializers
	-Wno-unused-parameter
)

target_include_directories(switch_fe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${LIBSLOWCONTROL_DIR})

target_link_libraries(switch_fe mfe midas rt pthread util libslowcontrol libmudaq dl)

if(DEBUG)
    add_definitions(-DMY_DEBUG)
endif()

install(TARGETS switch_fe DESTINATION bin)
