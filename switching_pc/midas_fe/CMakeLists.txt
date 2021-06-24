add_executable(switch_fe switch_fe.cpp)
#KB: disable most annoying compiler warnings that will not go away
target_compile_options(switch_fe PRIVATE
	-Wno-missing-field-initializers
	-Wno-unused-parameter
)

target_include_directories(switch_fe
        PRIVATE 
        ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${LIBSLOWCONTROL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include
        )

target_link_libraries(switch_fe mfe midas mscb rt pthread util libslowcontrol libmudaq dl)

if(DEBUG)
    add_definitions(-DMY_DEBUG)
endif()

install(TARGETS switch_fe DESTINATION bin)
