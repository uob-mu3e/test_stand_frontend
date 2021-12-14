FILE(GLOB_RECURSE INC_ALL "*.h")


include_directories(${LIBMUDAQ_DIR})
add_executable(rw rw.cpp ${INC_ALL})
add_executable(ddr3_test ddr3_test.cpp ${INC_ALL})
add_executable(swb_dmatest swb_dmatest.cpp ${INC_ALL})
add_executable(reset_link_a10_test reset_link_a10_test ${INC_ALL})
add_executable(test_slowcontrol test_slowcontrol ${INC_ALL})
target_link_libraries(rw libmudaq -lutil)
target_link_libraries(ddr3_test libmudaq -lutil)
target_link_libraries(swb_dmatest libmudaq -lutil)
target_link_libraries(reset_link_a10_test libmudaq -lutil)
target_link_libraries(test_slowcontrol libmudaq -lutil)
target_include_directories(test_slowcontrol PRIVATE ${LIBSLOWCONTROL_DIR})

install(TARGETS rw ddr3_test swb_dmatest reset_link_a10_test test_slowcontrol
        DESTINATION bin)
