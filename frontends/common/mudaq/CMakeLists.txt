
add_library(libmudaq STATIC
    mudaq_device.cpp mudaq_device.h utils.hpp     
    mudaq_circular_buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.h)
target_link_libraries(libmudaq  ${Boost_LIBRARIES})
