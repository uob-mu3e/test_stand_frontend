cmake_minimum_required(VERSION 3.0)

project(clockfrontend)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost 1.40.0 REQUIRED system)

add_library(ipbus "src/ipbus.cpp")
target_include_directories(ipbus PUBLIC "include")
target_include_directories(ipbus PUBLIC ${Boost_INCLUDE_DIRS})


add_library(clockboard "src/clockboard.cpp")
target_include_directories(clockboard PUBLIC "include")
target_link_libraries(clockboard ipbus)

add_executable(ipbustest "src/ipbustest.cpp")
target_include_directories(ipbustest PRIVATE "include")
target_link_libraries(ipbustest ipbus clockboard ${Boost_LIBRARIES})
