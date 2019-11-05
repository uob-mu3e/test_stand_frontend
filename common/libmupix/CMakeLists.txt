add_library(libmupix STATIC
    include/mupix_config.h
    include/mupix_MIDAS_config.h
    include/mupix_midasodb.h
    include/mupix_FEB.h
    #src/mupix_config.cpp
    src/mupix_MIDAS_config.cpp
    src/mupix_midasodb.cpp
    src/mupix_FEB.cpp
)
target_link_libraries(libmupix libmudaq ${Boost_LIBRARIES})
target_include_directories(libmupix
PUBLIC 
	"include"
PRIVATE
	${MIDASINCLUDE_DIR}
	${LIBMUDAQ_DIR}
)
