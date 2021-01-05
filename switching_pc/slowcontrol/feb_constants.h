/************************************************
* Register map header file 
* Automatically generated from /home/nberger/online/switching_pc/slowcontrol/../../fe_board/firmware/FEB_common/sc_registers.vhd
* On 2021-01-05T12:23:42.323172
************************************************/

#ifndef FEB_SC_REGISTERS__H 
#define FEB_SC_REGISTERS__H 


#define FEB_SC_ADDR_RANGE_HI		255
#define FEB_SC_ADDR_RANGE_LOW		0
#define GET_FEB_SC_ADDR_RANGE(REG) ((REG>>0)&0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff) 
#define SET_FEB_SC_ADDR_RANGE(REG, VAL) ((REG & (~(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff<< 0))) | ((VAL & 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff)<< 0))  
#define FEB_SC_DATA_SIZE_RANGE_HI		512
#define FEB_SC_DATA_SIZE_RANGE_LOW		1
#define GET_FEB_SC_DATA_SIZE_RANGE(REG) ((REG>>1)&0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff) 
#define SET_FEB_SC_DATA_SIZE_RANGE(REG, VAL) ((REG & (~(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff<< 1))) | ((VAL & 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff)<< 1))  
#define STATUS_REGISTER_R		0x00
#define GIT_HASH_REGISTER_R		0x01
#define FPGA_TYPE_REGISTER_R		0x02
#define FPGA_ID_REGISTER_RW		0x03
#define CMD_LEN_REGISTER_RW		0x04
#define CMD_OFFSET_REGISTER_RW		0x05
#define RUN_STATE_RESET_BYPASS_REGISTER_RW		0x06
#define RUN_STATE_RANGE_HI		31
#define RUN_STATE_RANGE_LOW		16
#define GET_RUN_STATE_RANGE(REG) ((REG>>16)&0xffff) 
#define SET_RUN_STATE_RANGE(REG, VAL) ((REG & (~(0xffff<< 16))) | ((VAL & 0xffff)<< 16))  
#define RESET_BYPASS_RANGE_HI		15
#define RESET_BYPASS_RANGE_LOW		0
#define GET_RESET_BYPASS_RANGE(REG) ((REG>>0)&0xffff) 
#define SET_RESET_BYPASS_RANGE(REG, VAL) ((REG & (~(0xffff<< 0))) | ((VAL & 0xffff)<< 0))  
#define RESET_OPTICAL_LINKS_REGISTER_RW		0x08
#define RESET_PHASE_REGISTER_R		0x09
#define MERGER_RATE_REGISTER_R		0x0a
#define ARRIA_TEMP_REGISTER_RW		0x10
#define MAX10_ADC_0_1_REGISTER_R		0x11
#define MAX10_ADC_2_3_REGISTER_R		0x12
#define MAX10_ADC_4_5_REGISTER_R		0x13
#define MAX10_ADC_6_7_REGISTER_R		0x14
#define MAX10_ADC_8_9_REGISTER_R		0x15
#define FIREFLY1_TEMP_REGISTER_R		0x16
#define FIREFLY1_VOLT_REGISTER_R		0x17
#define FIREFLY1_RX1_POW_REGISTER_R		0x18
#define FIREFLY1_RX2_POW_REGISTER_R		0x19
#define FIREFLY1_RX3_POW_REGISTER_R		0x1a
#define FIREFLY1_RX4_POW_REGISTER_R		0x1b
#define FIREFLY1_ALARM_REGISTER_R		0x1c
#define FIREFLY2_TEMP_REGISTER_R		0x1d
#define FIREFLY2_VOLT_REGISTER_R		0x1e
#define FIREFLY2_RX1_POW_REGISTER_R		0x1f
#define FIREFLY2_RX2_POW_REGISTER_R		0x20
#define FIREFLY2_RX3_POW_REGISTER_R		0x21
#define FIREFLY2_RX4_POW_REGISTER_R		0x22
#define REG_AREA_RANGE_HI		7
#define REG_AREA_RANGE_LOW		6
#define GET_REG_AREA_RANGE(REG) ((REG>>6)&0x3) 
#define SET_REG_AREA_RANGE(REG, VAL) ((REG & (~(0x3<< 6))) | ((VAL & 0x3)<< 6))  


#endif  //#ifndef FEB_SC_REGISTERS__H 
