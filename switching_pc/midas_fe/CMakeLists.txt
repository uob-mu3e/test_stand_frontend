add_executable(switch_fe switch_fe.cpp)
add_executable(tdac_test_file_generate generate_tdac_test.cpp)
add_executable(tdac_test_file_read read_tdac_test.cpp)
add_executable(ro_swb_fe ro_swb_fe.cpp ${INC_ALL})

target_include_directories(switch_fe
        PRIVATE 
        ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${LIBSLOWCONTROL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include
        ${MIDASSYS}/mscb/include
        )

target_include_directories(ro_swb_fe PRIVATE ${INCMUDAQ_DIR})

target_link_libraries(switch_fe mfe midas mscb rt pthread util libslowcontrol libmudaq dl)
target_link_libraries(ro_swb_fe mfe midas rt pthread util libmudaq -lpthread dl)

if(DEBUG)
    add_definitions(-DMY_DEBUG)
endif()

install(TARGETS switch_fe ro_swb_fe tdac_test_file_generate tdac_test_file_read DESTINATION bin)
