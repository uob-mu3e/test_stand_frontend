ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/feb_constants.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/feb_sc_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/feb_constants.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/feb_sc_registers.vhd
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/scifi_registers.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/scifi_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/scifi_registers.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/scifi_registers.vhd
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/a10_counters.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/a10_counters.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/a10_counters.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/a10_counters.vhd
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/mupix_registers.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/mupix_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/mupix_registers.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/mupix_registers.vhd
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/feb_sc_registers.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/makeheader.py ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/feb_sc_registers.vhd ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/feb_sc_registers.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/registers/feb_sc_registers.vhd
)

add_library(libslowcontrol STATIC
    FEBSlowcontrolInterface.cpp
    DummyFEBSlowcontrolInterface.cpp
    MuFEB.cpp feblist.cpp mappedfeb.cpp linkstatus.cpp
    mutrig/Mutrig_FEB.cpp mutrig/mutrig_midasodb.cpp mutrig/MutrigConfig.cpp mutrig/mutrig_MIDAS_config.cpp
    mupix/mupix_FEB.cpp mupix/mupix_midasodb.cpp mupix/mupix_config.cpp
    scifi/SciFi_FEB.cpp
    tiles/Tiles_FEB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/feb_constants.h #needs to be here explicitly for make rules to work!
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/feb_sc_registers.h #needs to be here explicitly for make rules to work!
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/scifi_registers.h #needs to be here explicitly for make rules to work!
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/mupix_registers.h #needs to be here explicitly for make rules to work!
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include/a10_counters.h #needs to be here explicitly for make rules to work!
)

target_include_directories(libslowcontrol PUBLIC . ./mupix ./mutrig ./tiles ./scifi ../../common/include ${CMAKE_SOURCE_DIR}/common/firmware/include)
target_link_libraries(libslowcontrol libmudaq libmubanks midas ${Boost_LIBRARIES})
