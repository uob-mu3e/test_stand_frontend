cmake_minimum_required(VERSION 3.0)

add_executable(config_mutrig_test config_mutrig.cpp ${INC_ALL})
add_executable(config_mutrig_test_new config_mutrig_new.cpp ${INC_ALL})
add_executable(config_mupix_test config_mupix.cpp ${INC_ALL})

target_include_directories(config_mutrig_test PRIVATE ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR})
target_include_directories(config_mutrig_test_new PRIVATE ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR})
target_include_directories(config_mupix_test PRIVATE ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUPIX_DIR})
target_link_libraries(config_mutrig_test libmudaq_scifi libmutrig)
target_link_libraries(config_mutrig_test_new libmudaq_scifi libmutrig)
target_link_libraries(config_mupix_test libmudaq_scifi libmupix)


install(TARGETS config_mutrig_test config_mutrig_test_new config_mupix_test DESTINATION unittests)
