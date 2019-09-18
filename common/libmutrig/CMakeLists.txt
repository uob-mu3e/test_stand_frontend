add_library(libmutrig STATIC
    include/mutrig_config.h
    include/mutrig_MIDAS_config.h
    include/mutrig_midasodb.h
    src/mutrig_config.cpp
    src/mutrig_MIDAS_config.cpp
    src/mutrig_midasodb.cpp
    include/SciFi_FEB.h
    src/SciFi_FEB.cpp
)
target_link_libraries(libmutrig  libmudaq ${Boost_LIBRARIES})
target_include_directories(libmutrig
PUBLIC 
	"include"
PRIVATE
	${MIDASINCLUDE_DIR}
	${LIBMUDAQ_DIR}
)
