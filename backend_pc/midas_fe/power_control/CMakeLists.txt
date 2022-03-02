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

add_executable(power_testbench
    power_testbench.cpp   GenesysDriver.cpp Keithley2611BDriver.cpp Keithley2450Driver.cpp    TCPClient.cpp
    HMP4040Driver.cpp  PowerDriver.cpp
    ${MIDASSYS}/drivers/class/multi.cxx
    ${MIDASSYS}/drivers/device/mscbdev.cxx
    ${MIDASSYS}/mscb/src/mscb.cxx
    )
target_include_directories(power_testbench
    PRIVATE
        ${MIDASSYS}/drivers
        ${MIDASSYS}/mscb/include)
target_link_libraries(power_testbench mfe midas rt dl pthread util ${Boost_LIBRARIES})


install(TARGETS power_testbench DESTINATION bin)

add_subdirectory(old_power_control)
