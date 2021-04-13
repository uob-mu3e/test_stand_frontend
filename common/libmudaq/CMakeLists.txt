ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/registers/a10_pcie_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/registers/a10_pcie_registers.vhd
)

ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../include/farm_constants.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/registers/a10_pcie_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../include/farm_constants.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/registers/a10_pcie_registers.vhd
)

add_library(libmudaq STATIC
    mudaq_device.cpp
    mudaq_dummy.cpp
    asic_config_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
    )
target_include_directories(libmudaq PUBLIC .)
target_link_libraries(libmudaq midas ${Boost_LIBRARIES})
