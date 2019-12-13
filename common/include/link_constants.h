#ifndef LINK_CONSTANTS_H
#define LINK_CONSTANTS_H

/* Maximum number of  swiutching boards */
const int MAX_N_SWITCHINGBOARDS = 4;

/* Maximum number of links per switching board */
const int MAX_LINKS_PER_SWITCHINGBOARD = 48;

/* Maximum number of frontenboards */
const int MAX_N_FRONTENDBOARDS = MAX_N_SWITCHINGBOARDS*MAX_LINKS_PER_SWITCHINGBOARD;

/* Identification of FEB by subsystem */
enum FEBTYPE {Undefined, Pixel, Fibre, Tile, FibresSecondary};
const std::string FEBTYPE_STR[5]={"Undefined","Pixel","Fibre","Tile", "FibreSecondary"};

/* Status of links */
enum LINKSTATUS {Disabled, OK, Unknown, Fault};

/* Masking of FEBs */
enum FEBLINKMASK {OFF, SCOn, DataOn, ON};


#endif // LINK_CONSTANTS_H
