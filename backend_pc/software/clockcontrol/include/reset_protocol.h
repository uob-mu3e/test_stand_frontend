#ifndef RESET_PROTOCOL_H
#define RESET_PROTOCOL_H

#include <stdint.h>
#include <map>
#include <string>

struct resetcommand{
    const uint8_t command;
    bool has_payload;
};

struct reset{
    const std::map<std::string, resetcommand> commands = { {"Run Prepare",     {0x10, true}},
                                                            {"Sync",            {0x11, false}},
                                                            {"Start Run",       {0x12, false}},
                                                            {"End Run",         {0x13, false}},
                                                            {"Abort Run",       {0x14, false}},
                                                            {"Start Link Test", {0x20, true}},
                                                            {"Stop Link Test",  {0x21, false}},
                                                            {"Start Sync Test", {0x24, true}},
                                                            {"Stop Sync Test",  {0x25, false}},
                                                            {"Test Sync",       {0x26, true}},
                                                            {"Reset",           {0x30, true}},
                                                            {"Stop Reset",      {0x31, true}},
                                                            {"Enable",          {0x32, false}},
                                                            {"Disable",         {0x33, false}},
                                                            {"Address",         {0x40, true}}
                                                          };



};


#endif // RESET_PROTOCOL_H
