#ifndef MUPIXBANKS_H
#define MUPIXBANKS_H

#include <array>
#include <string>
#include <string>

#include "odbxx.h"

#include "link_constants.h"

using std::array;
using std::string;
using std::to_string;

using midas::odb;



//// PSLL
/// TODO: we need this for different SWBs in the future for now its only central
constexpr int per_fe_PSLL_size = 4;
const int lvds_links_per_feb = 36;
const string banknamePSLL = "PSLL";

void create_psll_names_in_odb(odb & settings, int N_FEBS_MUPIX){
    int bankindex = 0;
    int N_LINKS = lvds_links_per_feb;
    string cntnamestr = banknamePSLL;

    for(uint32_t i=0; i < N_FEBS_MUPIX; i++){
        for(uint32_t j=0; j < N_LINKS; j++){
            string name = "FEB" + to_string(i);

            string * s = new string(name);
            (*s) += " LVDS ID";
            settings[cntnamestr][bankindex++] = s;

            s = new string(name);
            (*s) += " NUM HITS LVDS";
            settings[cntnamestr][bankindex++] = s;

            s = new string(name);
            (*s) += " NUM MP HITS LVDS";
            settings[cntnamestr][bankindex++] = s;

            s = new string(name);
            (*s) += " LVDS STATUS";
            settings[cntnamestr][bankindex++] = s;
        }
    }
}

#endif // MUPIXBANKS_H
