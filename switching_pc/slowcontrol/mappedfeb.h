#ifndef MAPPEDFEB_H
#define MAPPEDFEB_H

#include <stdint.h>
#include <string>

#include "link_constants.h"
#include "feb_constants.h"

class mappedFEB{
public:
    mappedFEB(uint16_t ID, uint32_t linkmask, std::string physName):LinkID(ID),mask(linkmask),fullname_link(physName){};
    bool IsScEnabled(){return mask&FEBLINKMASK::SCOn;}
    bool IsDataEnabled(){return mask&FEBLINKMASK::DataOn;}
    uint16_t GetLinkID(){return LinkID;}
    std::string GetLinkName(){return fullname_link;}
    //getters for FPGAPORT_ID and SB_ID (physical link address, independent on number of links per FEB)
    uint8_t SB_Number(){return LinkID/MAX_LINKS_PER_SWITCHINGBOARD;}
    uint8_t SB_Port()  {return LinkID%MAX_LINKS_PER_SWITCHINGBOARD;}
protected:
    uint16_t LinkID;	//global numbering. sb_id=LinkID/MAX_LINKS_PER_SWITCHINGBOARD, sb_port=LinkID%MAX_LINKS_PER_SWITCHINGBOARD
    uint32_t mask;
    std::string fullname_link;
};


#endif // MAPPEDFEB_H
