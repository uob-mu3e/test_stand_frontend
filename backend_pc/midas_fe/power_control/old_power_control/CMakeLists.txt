cmake_minimum_required(VERSION 3.0)
set_source_files_properties(power_supply_fe.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(keithley2611Bdev.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(hmp4040dev.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(lv.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(hv.c PROPERTIES LANGUAGE CXX )

add_executable(power_supply_fe power_supply_fe.c )
target_include_directories(power_supply_fe PRIVATE ${CMAKE_SOURCE_DIR}/common/include)
target_link_libraries(power_supply_fe keithley2611BdevLib mfe midas rt pthread util ${Boost_LIBRARIES})

add_library(keithley2611BdevLib STATIC lv.c hv.c hmp4040dev.c keithley2611Bdev.c)
