cmake_minimum_required(VERSION 3.0)

add_executable(netfe netfe.cpp)
add_executable(copyHostnames copyHostnames.cpp)

target_link_libraries(netfe mfe midas rt pthread util dl ${Boost_LIBRARIES})

target_link_libraries(copyHostnames rt pthread util ${Boost_LIBRARIES})

install(TARGETS netfe copyHostnames DESTINATION bin)
#install(TARGETS copyHostnames DESTINATION bin)
