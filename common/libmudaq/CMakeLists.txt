#ADD_CUSTOM_COMMAND(
  #OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.h
  #COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../firmware/pcie/source/mudaq_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mudaq_registers.h
  #COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.h
  #DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.vhd)

add_library(libmudaq STATIC
    mudaq_device.cpp mudaq_device.h utils.hpp     
    mudaq_circular_buffer.hpp
    asic_config_base.h
    asic_config_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.h ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h)
target_link_libraries(libmudaq  midas ${Boost_LIBRARIES})


add_library(libmudaq_scifi STATIC
    mudaq_device_scifi.cpp mudaq_device_scifi.h utils.hpp
    mudaq_circular_buffer.hpp
    asic_config_base.h
    asic_config_base.cpp
    MuFEB.cpp MuFEB.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/mudaq_registers.h ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h)
target_link_libraries(libmudaq_scifi  midas ${Boost_LIBRARIES})


target_include_directories(libmudaq_scifi
PUBLIC 
	"include"
PRIVATE
	${MIDASINCLUDE_DIR}
	${LIBMUDAQ_DIR}
	${INCMUDAQ_DIR}
)
