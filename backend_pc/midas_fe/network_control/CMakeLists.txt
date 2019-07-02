cmake_minimum_required(VERSION 3.0)

add_executable(netfe netfe.cpp ${MXML_DIR}/mxml.cxx)
add_executable(copyHostnames copyHostnames.cpp)

target_include_directories(netfe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR})
target_link_libraries(netfe mfe midas rt pthread util ${Boost_LIBRARIES})

target_include_directories(copyHostnames PRIVATE ${MIDAS_DIR}/include ${MXML_DIR})
target_link_libraries(copyHostnames rt pthread util ${Boost_LIBRARIES})

install(TARGETS netfe copyHostnames DESTINATION bin)
#install(TARGETS copyHostnames DESTINATION bin)
