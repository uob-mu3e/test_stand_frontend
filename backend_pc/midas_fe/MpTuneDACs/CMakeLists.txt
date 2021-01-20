cmake_minimum_required(VERSION 3.0)

add_executable(MpTuneDACsDB MpTuneDACsDB.cpp)

target_link_libraries(MpTuneDACsDB midas rt dl pthread util ${Boost_LIBRARIES})

install(TARGETS MpTuneDACsDB DESTINATION bin)
