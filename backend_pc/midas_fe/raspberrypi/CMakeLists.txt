cmake_minimum_required(VERSION 3.1)

add_executable(usb_clock_pi ucfe.cpp)

target_link_libraries(usb_clock_pi mfe midas rt pthread util ${Boost_LIBRARIES})

install(TARGETS usb_clock_pi DESTINATION bin)
