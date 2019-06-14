FILE(GLOB_RECURSE INC_ALL "*.h")

cuda_add_executable(farm_fe farm_fe.cu ${INC_ALL})
target_link_libraries(farm_fe -lcuda libmudaq -lutil)
target_link_libraries(farm_fe mfe midas rt pthread util)
target_include_directories(farm_fe PRIVATE ${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR})

install(TARGETS farm_fe
        DESTINATION bin)
