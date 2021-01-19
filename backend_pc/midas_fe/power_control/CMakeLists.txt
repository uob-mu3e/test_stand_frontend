cmake_minimum_required(VERSION 3.1)

file(GLOB SOURCES "*.cpp")
file(GLOB INC "*.h")

add_executable(power ${SOURCES})
target_include_directories(power PRIVATE ${MIDAS_DIR}/include ${MIDAS_DIR}/drivers ${MXML_DIR} ${MIDASINCLUDE_DIR} ${INC_PATH} ${INC})
target_link_libraries(power mfe midas rt dl pthread util ${Boost_LIBRARIES})

install(TARGETS power DESTINATION bin)

add_subdirectory(old_power_control)
