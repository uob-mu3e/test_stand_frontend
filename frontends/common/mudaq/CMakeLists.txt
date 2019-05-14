ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mudaq_registers.h
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../firmware/common/scripts/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../firmware/pcie/source/mudaq_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mudaq_registers.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../firmware/pcie/source/mudaq_registers.vhd)

add_library(libmudaq STATIC
    mudaq_device.cpp mudaq_device.h utils.hpp     
    mudaq_circular_buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mudaq_registers.h)
target_link_libraries(libmudaq  ${Boost_LIBRARIES})
