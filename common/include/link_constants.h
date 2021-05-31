#ifndef LINK_CONSTANTS_H
#define LINK_CONSTANTS_H

#include<string>

/* Maximum number of  switching boards */
const uint32_t MAX_N_SWITCHINGBOARDS = 4;

/* Maximum number of links per switching board */
const uint32_t MAX_LINKS_PER_SWITCHINGBOARD = 48;

/* Maximum number of FEBs per switching board */
const uint32_t MAX_FEBS_PER_SWITCHINGBOARD = 34;

/* Maximum number of frontenboards */
const uint32_t MAX_N_FRONTENDBOARDS = MAX_N_SWITCHINGBOARDS*MAX_LINKS_PER_SWITCHINGBOARD;

/* Number of FEBs in final system */
constexpr uint32_t N_FEBS[MAX_N_SWITCHINGBOARDS] = {34, 33, 33, 12};

constexpr uint32_t N_FEBS_TOTAL = N_FEBS[0]+N_FEBS[1]+N_FEBS[2]+N_FEBS[3];

/* Number of crates in final system */
constexpr uint32_t N_FEBCRATES = 8;

/* Maximum number of FEBS in a crate */
constexpr uint32_t MAX_FEBS_PER_CRATE = 16;

/* Identification of FEB by subsystem */
enum FEBTYPE {Undefined, Pixel, Fibre, Tile, FibreSecondary};
const std::string FEBTYPE_STR[5]={"Undefined","Pixel","Fibre","Tile", "FibreSecondary"};

/* Status of links */
enum LINKSTATUS {Disabled, OK, Unknown, Fault};

/* Masking of FEBs */
enum FEBLINKMASK {OFF, SCOn, DataOn, ON};


#endif // LINK_CONSTANTS_H
