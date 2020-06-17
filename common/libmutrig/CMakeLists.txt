add_library(libmutrig STATIC
    include/MutrigConfig.h
    include/mutrig_MIDAS_config.h
    include/mutrig_midasodb.h
    src/MutrigConfig.cpp
    src/mutrig_MIDAS_config.cpp
    src/mutrig_midasodb.cpp
    include/Mutrig_FEB.h
    src/Mutrig_FEB.cpp
    include/SciFi_FEB.h
    src/SciFi_FEB.cpp
    include/Tiles_FEB.h
    src/Tiles_FEB.cpp
)
target_link_libraries(libmutrig  libmudaq dl ${Boost_LIBRARIES})
target_include_directories(libmutrig
PUBLIC
	"include"
PRIVATE
	${MIDASINCLUDE_DIR}
	${LIBMUDAQ_DIR}
	${INCMUDAQ_DIR}
)
