FILE(GLOB_RECURSE INC_ALL "*.h")


include_directories(${LIBMUDAQ_DIR})
add_executable(rw rw.cpp ${INC_ALL})
add_executable(ddr3_test ddr3_test.cpp ${INC_ALL})
add_executable(swb_dmatest swb_dmatest.cpp ${INC_ALL})
target_link_libraries(rw libmudaq -lutil)
target_link_libraries(ddr3_test libmudaq -lutil)
target_link_libraries(swb_dmatest libmudaq -lutil)

install(TARGETS rw ddr3_test swb_dmatest
        DESTINATION bin)
