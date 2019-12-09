cmake_minimum_required(VERSION 3.0)


add_library(ipbus "src/ipbus.cpp")
target_include_directories(ipbus PUBLIC "include")
target_include_directories(ipbus PUBLIC ${Boost_INCLUDE_DIRS})

install(TARGETS ipbus DESTINATION lib)

add_library(clockboard "src/clockboard.cpp")
target_include_directories(clockboard PUBLIC "include")
target_link_libraries(clockboard ipbus)

install(TARGETS clockboard DESTINATION lib)

add_executable(ipbustest "src/ipbustest.cpp")
target_include_directories(ipbustest PRIVATE "include")
target_link_libraries(ipbustest ipbus clockboard ${Boost_LIBRARIES})
message(${Boost_LIBRARIES}) 
install(TARGETS ipbustest DESTINATION bin)

