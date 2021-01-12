ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/feb_constants.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../fe_board/firmware/FEB_common/sc_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/feb_constants.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../fe_board/firmware/FEB_common/sc_registers.vhd
)

add_library(libslowcontrol STATIC
    FEB_slowcontrol.cpp feb_constants.h
)
target_include_directories(libslowcontrol PUBLIC . ./mupix ./mutrig ./tiles ./scifi)
target_link_libraries(libslowcontrol libmudaq midas ${Boost_LIBRARIES})
