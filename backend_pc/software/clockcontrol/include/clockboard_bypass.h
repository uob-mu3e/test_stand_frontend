#ifndef CLOCKBOARD_BYPASS_H
#define CLOCKBOARD_BYPASS_H

#include "ipbus.h"
#include "reset_protocol.h"
#include "clockboard.h"
class clockboard_bypass:public clockboard
{
public:
    clockboard_bypass(const char * addr, int port):clockboard(addr,port){}
    bool isConnected(){return true;}

    int init_clockboard(uint16_t clkinvert = 0x0A00, uint16_t rstinvert= 0x0008, uint16_t clkdisable = 0x0AA, uint16_t rstdisable = 0xAA0){return 1;}
    int map_daughter_fibre(uint8_t daughter_num, uint16_t fibre_num){return 1;}
    // Write "reset" commands
    int write_command(uint8_t command, uint32_t payload =0, bool has_payload = false){
        //printf("write_command(%2.2x,%8.8x,%s)\n",command,payload,has_payload?"true":"false");
	DWORD val=0xbcbcbcbc;
	//write payload ODB - Switching frontend will send this to FEB
	if(has_payload)
		val=payload;
        db_set_value(hDB,0,"/Equipment/Switching/Settings/Reset Bypass Payload", &val, sizeof(uint32_t), 1, TID_DWORD);
	//write reset char ODB - Switching frontend will send this to FEB
	val=(1<<8) | command;
        db_set_value(hDB,0,"/Equipment/Switching/Settings/Reset Bypass Command", &val, sizeof(uint32_t), 1, TID_DWORD);
	usleep(100000);
	//wait for flag to be resetted
	INT valsiz=sizeof(uint32_t);
	int timeout_cnt=500;
	do{
		db_get_value(hDB,0,"/Equipment/Switching/Settings/Reset Bypass Command", &val, &valsiz, TID_DWORD,false);
		printf("%d: %2.2x\n",timeout_cnt,val);
		if((val&0xff) == 0) break;
		usleep(100000);
	}while(--timeout_cnt>0);
	if(timeout_cnt==0){
		cm_msg(MERROR, "clockboard_bypass::write_command", "timeout waiting for odb flag reset. Switching FE running?");
		return -1;
	}
	return SUCCESS;
    }

    int write_command(const char * name, uint32_t payload =0, uint16_t address =0){
        //printf("write_command(%s,%8.8x,%u)\n",name,payload,address);
        //cm_msg(MINFO, "clockboard_bypass::write_command", "sending %s",name);
        auto it = reset_protocol.commands.find(name);
        if(it != reset_protocol.commands.end()){
           if(address==0){
              return write_command(it->second.command, payload, it->second.has_payload);
           }else{
              // addressed command
	      printf("Addressed command not supported in bypass. Command: %s\n",name);
              return 0;
           }
        }
        printf("Unknown command %s\n",name);
        return -1;
    };

    // I2C interface
//    int init_i2c();
//    int read_i2c(uint8_t dev_addr, uint8_t &data);
//    int read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t &data);
//    int read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[]);

//    int write_i2c(uint8_t dev_addr, uint8_t data);
//    int write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data);
//    int write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[]);

    // SI3545 programming - note that this uses the register map
    // generated with the clock builder tool
//    int load_SI3545_reg_map();
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

//    vector<uint8_t> read_rx_amplitude();
//    vector<uint8_t> read_rx_emphasis();

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
//    int enable_daughter_12c(uint8_t dev_addr, uint8_t i2c_bus_num);
//    int disable_daughter_12c(uint8_t dev_addr);

    float read_daughter_board_current(uint8_t daughter){return 0;};
    float read_mother_board_current(){return 0;}

    float read_daughter_board_voltage(uint8_t daughter){return 0;}
    float read_mother_board_voltage(){return 0;}
    
    float read_fan_current(){return 0;}

//    int configure_daughter_current_monitor(uint8_t daughter, uint16_t config);
//    int configure_mother_current_monitor(uint16_t config);

public:
    reset reset_protocol;
};

#endif // CLOCKBOARD_BYPASS_H
