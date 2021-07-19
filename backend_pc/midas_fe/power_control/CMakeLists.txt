add_executable(power
    GenesysDriver.cpp  power.cpp        TCPClient.cpp
    HMP4040Driver.cpp  PowerDriver.cpp
    ${MIDASSYS}/drivers/class/multi.cxx
    ${MIDASSYS}/drivers/device/mscbdev.cxx
    ${MIDASSYS}/mscb/src/mscb.cxx
    )
target_include_directories(power
    PRIVATE
        ${MIDASSYS}/drivers
        ${MIDASSYS}/mscb/include)
target_link_libraries(power mfe midas rt dl pthread util ${Boost_LIBRARIES})


install(TARGETS power DESTINATION bin)

add_subdirectory(old_power_control)
