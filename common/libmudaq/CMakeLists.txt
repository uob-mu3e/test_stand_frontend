ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/assignments/mudaq_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/assignments/mudaq_registers.vhd
)

ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../include/farm_constants.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/assignments/mudaq_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../include/farm_constants.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/assignments/mudaq_registers.vhd
)

#ADD_CUSTOM_COMMAND(
#  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../include/feb_constants.h
#  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../fe_board/firmware/FEB_common/sc_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../include/feb_constants.h
#  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../fe_board/firmware/FEB_common/sc_registers.vhd
#)


add_library(libmudaq STATIC
    mudaq_device.cpp #mudaq_device.h utils.hpp
    mudaq_dummy.cpp #mudaq_dummy.h
    #mudaq_circular_buffer.hpp
    #asic_config_base.h
    asic_config_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
    )
target_include_directories(libmudaq PUBLIC .)
target_link_libraries(libmudaq  midas ${Boost_LIBRARIES})


#add_library(libmudaq_scifi STATIC
#    mudaq_device_scifi.cpp mudaq_device_scifi.h utils.hpp
#    mudaq_circular_buffer.hpp
#    asic_config_base.h
#    asic_config_base.cpp
#    MuFEB.cpp MuFEB.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/../include/feb_constants.h
#    )
#target_link_libraries(libmudaq_scifi  midas ${Boost_LIBRARIES})

#add_library(libmudaq_test STATIC
#		mudaq_dummy.cpp mudaq_dummy.h utils.hpp
#		mudaq_device.cpp mudaq_device.h
#		mudaq_circular_buffer.hpp
#		asic_config_base.h
#		asic_config_base.cpp
#		${CMAKE_CURRENT_SOURCE_DIR}/../include/switching_constants.h
#		)
#target_link_libraries(libmudaq_test midas ${Boost_LIBRARIES})


#target_include_directories(libmudaq_scifi
#PUBLIC
#	"include"
#PRIVATE
#	${LIBMUDAQ_DIR}
#	${INCMUDAQ_DIR}
#)
