add_executable(switch_fe switch_fe.cpp)
add_executable(tdac_test_file_generate generate_tdac_test.cpp)
add_executable(tdac_test_file_read read_tdac_test.cpp)

target_include_directories(switch_fe
        PRIVATE 
        ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${LIBSLOWCONTROL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include
        ${MIDASSYS}/mscb/include
        )

target_link_libraries(switch_fe mfe midas mscb rt pthread util libslowcontrol libmudaq dl)

if(DEBUG)
    add_definitions(-DMY_DEBUG)
endif()

install(TARGETS switch_fe tdac_test_file_generate tdac_test_file_read DESTINATION bin)
