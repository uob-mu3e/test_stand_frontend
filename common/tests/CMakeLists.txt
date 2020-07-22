cmake_minimum_required(VERSION 3.0)

add_executable(config_mutrig_test config_mutrig.cpp ${INC_ALL})
add_executable(config_mupix_test config_mupix.cpp ${INC_ALL})
add_executable(test_odb test_odb.cpp ${INC_ALL})

target_include_directories(config_mutrig_test PRIVATE ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR})
target_include_directories(config_mupix_test PRIVATE ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUPIX_DIR})
target_include_directories(test_odb PRIVATE ${INCMUDAQ_DIR} ${MIDAS_DIR}/include ${MIDASINCLUDE_DIR})

target_link_libraries(config_mutrig_test libmudaq_scifi libmutrig)
target_link_libraries(config_mupix_test libmudaq_scifi libmupix)
target_link_libraries(test_odb mfe midas rt pthread util dl)

install(TARGETS config_mutrig_test config_mupix_test test_odb DESTINATION unittests)
