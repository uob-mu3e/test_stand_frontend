add_executable(switch_fe switch_fe.cpp)

target_include_directories(switch_fe
        PRIVATE 
        ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${LIBSLOWCONTROL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include
        ${MIDASSYS}/mscb/include
        )

target_link_libraries(switch_fe mfe midas mscb rt pthread util libslowcontrol libmudaq dl)

if(DEBUG)
    add_definitions(-DMY_DEBUG)
endif()

install(TARGETS switch_fe DESTINATION bin)
