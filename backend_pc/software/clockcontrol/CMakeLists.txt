add_library(ipbus "src/ipbus.cpp")
target_include_directories(ipbus PUBLIC "include")
target_include_directories(ipbus PUBLIC ${Boost_INCLUDE_DIRS})

install(TARGETS ipbus DESTINATION lib)

add_library(clockboard "src/clockboard.cpp" "src/clockboard_a10.cpp")
target_include_directories(clockboard PUBLIC "include")
target_include_directories(clockboard PRIVATE ${CMAKE_SOURCE_DIR}/common/include ${CMAKE_SOURCE_DIR}/common/libmudaq)
target_link_libraries(clockboard ipbus libmudaq)

install(TARGETS clockboard DESTINATION lib)

add_executable(ipbustest "src/ipbustest.cpp")
target_include_directories(ipbustest PRIVATE "include" ${CMAKE_SOURCE_DIR}/common/include)
target_link_libraries(ipbustest ipbus clockboard ${Boost_LIBRARIES})
message(${Boost_LIBRARIES}) 
install(TARGETS ipbustest DESTINATION bin)

