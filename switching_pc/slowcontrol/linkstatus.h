#ifndef LINKSTATUS_H
#define LINKSTATUS_H

#include <stdint.h>
#include <string>

#include "link_constants.h"

class LinkStatus{
public:
    LinkStatus():status(LINKSTATUS::Unknown), badmessagecounter(0), goodmessagecounter(0){};
    void SetStatus(uint32_t _status){status = _status;}
    uint32_t GetStatus(){return status;}
    bool LinkIsOK(){return status == LINKSTATUS::OK;}
    uint32_t GetGoodMessages(){return goodmessagecounter;}
    uint32_t GetBadMessages(){return badmessagecounter;}
    uint32_t CountGoodMessage(){return ++goodmessagecounter;}
    uint32_t CountBadMessage(){return ++badmessagecounter;}
    void ResetMessageCounters(){goodmessagecounter =0; badmessagecounter =0;}

protected:
    uint32_t status;
    uint32_t badmessagecounter;
    uint32_t goodmessagecounter;
};

#endif