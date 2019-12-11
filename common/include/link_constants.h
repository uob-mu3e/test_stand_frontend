#ifndef LINK_CONSTANTS_H
#define LINK_CONSTANTS_H

/* Maximum number of  swiutching boards */
const int MAX_N_SWITCHINGBOARDS = 4;

/* Maximum number of frontenboards */
const int MAX_N_FRONTENBOARDS = MAX_N_SWITCHINGBOARDS*48;

/* Identification of FEB by subsystem */
enum FEBTYPE {Undefined, Pixel, Fibre, Tile};
/* Status of links */
enum LINKSTATUS {Disabled, OK, Unknown, Fault};


#endif // LINK_CONSTANTS_H
