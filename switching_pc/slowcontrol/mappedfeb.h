#ifndef MAPPEDFEB_H
#define MAPPEDFEB_H

#include <stdint.h>
#include <string>

#include "link_constants.h"
#include "feb_constants.h"

class mappedFEB{
public:
    mappedFEB(uint16_t ID, uint32_t linkmask, std::string physName, uint16_t _crate, uint16_t _slot, uint32_t _version =20):
        LinkID(ID),mask(linkmask),fullname_link(physName),crate(_crate),slot(_slot),version(_version){};
    bool IsScEnabled() const {return mask&FEBLINKMASK::SCOn;}
    bool IsDataEnabled() const {return mask&FEBLINKMASK::DataOn;}
    uint16_t GetLinkID() const {return LinkID;}
    std::string GetLinkName() const {return fullname_link;}
    //getters for FPGAPORT_ID and SB_ID (physical link address, independent on number of links per FEB)
    uint8_t SB_Number() const {return LinkID/MAX_LINKS_PER_SWITCHINGBOARD;}
    uint8_t SB_Port() const {return LinkID%MAX_LINKS_PER_SWITCHINGBOARD;}
    uint32_t GetVersion() const {return version;}
    uint16_t GetCrate() const {return crate;}
    uint16_t GetSlot() const {return slot;}
protected:
    uint16_t LinkID;	//global numbering. sb_id=LinkID/MAX_LINKS_PER_SWITCHINGBOARD, sb_port=LinkID%MAX_LINKS_PER_SWITCHINGBOARD
    uint32_t mask;
    std::string fullname_link;
    uint32_t version;
    uint16_t crate;
    uint16_t slot;
};


#endif // MAPPEDFEB_H
