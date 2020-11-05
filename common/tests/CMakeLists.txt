cmake_minimum_required(VERSION 3.0)

add_subdirectory("${PROJECT_SOURCE_DIR}/modules/googletest" "modules/googletest")

# make cache clean
mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

#use this for cleaning folder with IDEs
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN} ${INC_ALL})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main mfe midas rt pthread util libmudaq_scifi libmutrig libmupix libmudaq_test dl)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    target_include_directories(${TESTNAME} PRIVATE ${LIBMUDAQ_DIR} ${INCMUDAQ_DIR} ${INCMUTRIG_DIR})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

macro(add_vhdl_test TESTNAME)
    configure_file(${PROJECT_SOURCE_DIR}/common/tests/vhdl_tests/${TESTNAME}/run_${TESTNAME}.sh vhdl_tests/${TESTNAME}/run_${TESTNAME}.sh @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vhdl_tests/${TESTNAME}/run_${TESTNAME}.sh DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endmacro()

package_add_test(config_mutrig_test config_mutrig.cpp)
package_add_test(config_mupix_test config_mupix.cpp)
package_add_test(test_odb test_odb.cpp)
package_add_test(test_mudaq_device test_mudaq_device.cpp)



add_vhdl_test(example_test)

install(TARGETS config_mutrig_test config_mupix_test test_odb test_mudaq_device DESTINATION bin)
