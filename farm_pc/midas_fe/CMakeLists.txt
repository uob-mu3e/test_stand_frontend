FILE(GLOB_RECURSE INC_ALL "*.h")

include_directories(${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})

add_executable(farm_fe farm_fe.cpp ${INC_ALL})
message(${INCMUDAQ_DIR})
target_include_directories(farm_fe PRIVATE ${INCMUDAQ_DIR})
target_link_libraries(farm_fe libmudaq dl)
target_link_libraries(farm_fe mfe midas rt pthread util)

install(TARGETS farm_fe
        DESTINATION bin)
