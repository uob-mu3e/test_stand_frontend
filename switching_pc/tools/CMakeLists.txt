add_executable(test_slowcontrol test_slowcontrol)

target_link_libraries(test_slowcontrol libmudaq -lutil)
target_include_directories(test_slowcontrol PRIVATE ${LIBSLOWCONTROL_DIR} ../../common/include)

install(TARGETS test_slowcontrol
        DESTINATION bin)