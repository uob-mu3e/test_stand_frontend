add_executable(env_control environment_control.cpp
    ${MIDASSYS}/drivers/class/multi
    ${MIDASSYS}/drivers/device/mscbdev
    ${MIDASSYS}/mscb/src/mscb
    ${MIDASSYS}/drivers/class/hv
    )
target_include_directories(env_control
    PRIVATE
        ${MIDASSYS}/drivers
        ${MIDASSYS}/mscb/include
    )
target_link_libraries(env_control mfe midas rt dl pthread util ${Boost_LIBRARIES})


install(TARGETS env_control DESTINATION bin)
