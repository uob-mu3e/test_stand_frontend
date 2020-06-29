/********************************************************************\

  Name:         MpTuneDACsDB.cpp
  Created by:   Martin Mueller

  Contents:     Store/read/write Mupix pixel tdacs outside of odb (first version)

\********************************************************************/

#include <string>
#include <iostream>
#include <array>
#include <functional>

#include "midas.h"
#include "odbxx.h"

/*------------------------------------------------------------------*/

int main() {

    cm_connect_experiment(NULL, NULL, "MpTDACsDB", NULL);
    //midas::odb::set_debug(true);

   // create ODB structure...
    midas::odb o = {
        {"chipIDreq", 0},
        {"chipIDactual", 0},
        {"store",false},
        {"loadrecent",false},
        {"loadnumber",false},
        {"runnumber",0},
        {"col0", std::array<short, 200>{} },
        {"col1", std::array<short, 200>{}},
        {"col2", std::array<short, 200>{}},
        {"col3", std::array<short, 200>{}},
        {"col4", std::array<short, 200>{}},
        {"col5", std::array<short, 200>{}},
        {"col6", std::array<short, 200>{}},
        {"col7", std::array<short, 200>{}},
        {"col8", std::array<short, 200>{}},
        {"col9", std::array<short, 200>{}},
        {"col10", std::array<short, 200>{}},
        {"col11", std::array<short, 200>{}},
        {"col12", std::array<short, 200>{}},
        {"col13", std::array<short, 200>{}},
        {"col14", std::array<short, 200>{}},
        {"col15", std::array<short, 200>{}},
        {"col16", std::array<short, 200>{}},
        {"col17", std::array<short, 200>{}},
        {"col18", std::array<short, 200>{}},
        {"col19", std::array<short, 200>{}},
        {"col20", std::array<short, 200>{}},
        {"col21", std::array<short, 200>{}},
        {"col22", std::array<short, 200>{}},
        {"col23", std::array<short, 200>{}},
        {"col24", std::array<short, 200>{}},
        {"col25", std::array<short, 200>{}},
        {"col26", std::array<short, 200>{}},
        {"col27", std::array<short, 200>{}},
        {"col28", std::array<short, 200>{}},
        {"col29", std::array<short, 200>{}},
        {"col30", std::array<short, 200>{}},
        {"col31", std::array<short, 200>{}},
        {"col32", std::array<short, 200>{}},
        {"col33", std::array<short, 200>{}},
        {"col34", std::array<short, 200>{}},
        {"col35", std::array<short, 200>{}},
        {"col36", std::array<short, 200>{}},
        {"col37", std::array<short, 200>{}},
        {"col38", std::array<short, 200>{}},
        {"col39", std::array<short, 200>{}},
        {"col40", std::array<short, 200>{}},
        {"col41", std::array<short, 200>{}},
        {"col42", std::array<short, 200>{}},
        {"col43", std::array<short, 200>{}},
        {"col44", std::array<short, 200>{}},
        {"col45", std::array<short, 200>{}},
        {"col46", std::array<short, 200>{}},
        {"col47", std::array<short, 200>{}},
        {"col48", std::array<short, 200>{}},
        {"col49", std::array<short, 200>{}},
        {"col50", std::array<short, 200>{}},
        {"col51", std::array<short, 200>{}},
        {"col52", std::array<short, 200>{}},
        {"col53", std::array<short, 200>{}},
        {"col54", std::array<short, 200>{}},
        {"col55", std::array<short, 200>{}},
        {"col56", std::array<short, 200>{}},
        {"col57", std::array<short, 200>{}},
        {"col58", std::array<short, 200>{}},
        {"col59", std::array<short, 200>{}},
        {"col60", std::array<short, 200>{}},
        {"col61", std::array<short, 200>{}},
        {"col62", std::array<short, 200>{}},
        {"col63", std::array<short, 200>{}},
        {"col64", std::array<short, 200>{}},
        {"col65", std::array<short, 200>{}},
        {"col66", std::array<short, 200>{}},
        {"col67", std::array<short, 200>{}},
        {"col68", std::array<short, 200>{}},
        {"col69", std::array<short, 200>{}},
        {"col70", std::array<short, 200>{}},
        {"col71", std::array<short, 200>{}},
        {"col72", std::array<short, 200>{}},
        {"col73", std::array<short, 200>{}},
        {"col74", std::array<short, 200>{}},
        {"col75", std::array<short, 200>{}},
        {"col76", std::array<short, 200>{}},
        {"col77", std::array<short, 200>{}},
        {"col78", std::array<short, 200>{}},
        {"col79", std::array<short, 200>{}},
        {"col80", std::array<short, 200>{}},
        {"col81", std::array<short, 200>{}},
        {"col82", std::array<short, 200>{}},
        {"col83", std::array<short, 200>{}},
        {"col84", std::array<short, 200>{}},
        {"col85", std::array<short, 200>{}},
        {"col86", std::array<short, 200>{}},
        {"col87", std::array<short, 200>{}},
        {"col88", std::array<short, 200>{}},
        {"col89", std::array<short, 200>{}},
        {"col90", std::array<short, 200>{}},
        {"col91", std::array<short, 200>{}},
        {"col92", std::array<short, 200>{}},
        {"col93", std::array<short, 200>{}},
        {"col94", std::array<short, 200>{}},
        {"col95", std::array<short, 200>{}},
        {"col96", std::array<short, 200>{}},
        {"col97", std::array<short, 200>{}},
        {"col98", std::array<short, 200>{}},
        {"col99", std::array<short, 200>{}},
        {"col100", std::array<short, 200>{}},
        {"col101", std::array<short, 200>{}},
        {"col102", std::array<short, 200>{}},
        {"col103", std::array<short, 200>{}},
        {"col104", std::array<short, 200>{}},
        {"col105", std::array<short, 200>{}},
        {"col106", std::array<short, 200>{}},
        {"col107", std::array<short, 200>{}},
        {"col108", std::array<short, 200>{}},
        {"col109", std::array<short, 200>{}},
        {"col110", std::array<short, 200>{}},
        {"col111", std::array<short, 200>{}},
        {"col112", std::array<short, 200>{}},
        {"col113", std::array<short, 200>{}},
        {"col114", std::array<short, 200>{}},
        {"col115", std::array<short, 200>{}},
        {"col116", std::array<short, 200>{}},
        {"col117", std::array<short, 200>{}},
        {"col118", std::array<short, 200>{}},
        {"col119", std::array<short, 200>{}},
        {"col120", std::array<short, 200>{}},
        {"col121", std::array<short, 200>{}},
        {"col122", std::array<short, 200>{}},
        {"col123", std::array<short, 200>{}},
        {"col124", std::array<short, 200>{}},
        {"col125", std::array<short, 200>{}},
        {"col126", std::array<short, 200>{}},
        {"col127", std::array<short, 200>{}}
    };

    o.connect("/Equipment/MuPix/TDACs");

    // watch ODB key for any change with lambda function
    midas::odb ow("/Equipment/Mupix/TDACs");
    ow.watch([](midas::odb &o) {
        std::cout << "Value of key \"" + o.get_full_path() + "\" changed to " << o << std::endl;
    });

    do {
        int status = cm_yield(100);
        if (status == SS_ABORT || status == RPC_SHUTDOWN)
            break;
    } while (!ss_kbhit());

    cm_disconnect_experiment();
    return 1;
}
