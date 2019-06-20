cmake_minimum_required(VERSION 3.0)

add_executable(crfe crfe.cpp ${MXML_DIR}/mxml.cxx)

target_include_directories(crfe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR})
target_link_libraries(crfe mfe midas rt pthread util clockboard ${Boost_LIBRARIES})


install(TARGETS crfe DESTINATION bin)
