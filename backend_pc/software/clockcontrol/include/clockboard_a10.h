#ifndef CLOCKBOARD_A10_H
#define CLOCKBOARD_A10_H

#include "odbxx.h"
#include "ipbus.h"
#include "reset_protocol.h"
#include "clockboard.h"

#include "mudaq_device.h"

using midas::odb;

class clockboard_a10:public clockboard
{
public:
    clockboard_a10(std::string addr, int port);
    bool isConnected();

    int init_clockboard(uint16_t clkinvert = 0x0A00, uint16_t rstinvert= 0x0008, uint16_t clkdisable = 0x0AA, uint16_t rstdisable = 0xAA0);
    int map_daughter_fibre(uint8_t daughter_num, uint16_t fibre_num){return 1;}
    // Write "reset" commands
    int write_command(uint8_t command, uint32_t payload =0, bool has_payload = false);
    int write_command(const char * name, uint32_t payload =0, uint16_t address =0);

    // Firefly interface
    bool firefly_present(uint8_t daughter, uint8_t index){return false;};
    uint16_t read_disabled_tx_clk_channels(){return 0;}
    int disable_tx_clk_channels(uint16_t channels){return 1;}
    uint16_t read_inverted_tx_clk_channels(){return 0;}
    int invert_tx_clk_channels(uint16_t channels){return 1;}

    uint16_t read_disabled_tx_rst_channels(){return 0;}
    int disable_tx_rst_channels(uint16_t channels){return 1;}
    uint16_t read_inverted_tx_rst_channels(){return 0;}
    int invert_tx_rst_channels(uint16_t channels){return 1;}

    int disable_rx_channels(uint16_t channelmask){return 1;}
    uint16_t read_disabled_rx_channels(){return 0;}

    int set_rx_amplitude(uint8_t amplitude){return 1;}
    int set_rx_emphasis(uint8_t emphasis){return 1;}

    float read_rx_firefly_temp(){return 0;}
    float read_rx_firefly_voltage(){return 0;}
    uint16_t read_rx_firefly_los(){return 0;}
    uint16_t read_rx_firefly_alarms(){return 0;}

    float read_tx_clk_firefly_temp(){return 0;}
    float read_tx_rst_firefly_temp(){return 0;}
    float read_tx_clk_firefly_voltage(){return 0;}
    float read_tx_rst_firefly_voltage(){return 0;}
    uint16_t read_tx_clk_firefly_lf(){return 0;}
    uint16_t read_tx_clk_firefly_alarms(){return 0;}
    uint16_t read_tx_rst_firefly_lf(){return 0;}
    uint16_t read_tx_rst_firefly_alarms(){return 0;}

    float read_tx_firefly_temp(uint8_t daughter, uint8_t index){return 0;}
    float read_tx_firefly_voltage(uint8_t daughter, uint8_t index){return 0;}

    int disable_tx_channels(uint8_t daughter, uint8_t firefly, uint16_t channelmask){return 1;}

    uint16_t read_tx_firefly_lf(uint8_t daughter, uint8_t index){return 0;}
    uint16_t read_tx_firefly_alarms(uint8_t daughter, uint8_t index){return 0;}
    // Mother and daughter card monitoring
    bool daughter_present(uint8_t daughter){return false;}
    uint8_t daughters_present(){return 0;}

    float read_daughter_board_current(uint8_t daughter){return 0;};
    float read_mother_board_current(){return 0;}

    float read_daughter_board_voltage(uint8_t daughter){return 0;}
    float read_mother_board_voltage(){return 0;}
    
    float read_fan_current(){return 0;}

protected:
    bool connected;
    mudaq::DmaMudaqDevice mu;

public:
    reset reset_protocol;
};

#endif // CLOCKBOARD_A10_H
