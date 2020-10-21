cmake_minimum_required(VERSION 3.0)

add_executable(crfe crfe.cpp)

target_include_directories(crfe PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${CMAKE_SOURCE_DIR}/common/include)
target_link_libraries(crfe mfe midas rt dl pthread util clockboard ${Boost_LIBRARIES})



install(TARGETS crfe DESTINATION bin)
