/************************************************
* Register map header file 
* Copied from automatically generated file, this needs to be implemented in the build flow
* For now hard coded
* On 2019-06-20T10:31:04.680970
************************************************/

#ifndef FEB_REGISTERS__H 
#define FEB_REGISTERS__H 


#define FE_DUMMYCTRL_REG	0x10
#define GET_FE_DUMMYCTRL_BIT_SPI(REG) ((REG>>0)& 0x1) 
#define SET_FE_DUMMYCTRL_BIT_SPI(REG) ((1<<0)| REG) 
#define UNSET_FE_DUMMYCTRL_BIT_SPI(REG) ((~(1<<0)) & REG) 

#define GET_FE_DUMMYCTRL_BIT_DATAGEN(REG) ((REG>>1)& 0x1) 
#define SET_FE_DUMMYCTRL_BIT_DATAGEN(REG) ((1<<1)| REG) 
#define UNSET_FE_DUMMYCTRL_BIT_DATAGEN(REG) ((~(1<<1)) & REG) 

#define GET_FE_DUMMYCTRL_BIT_SHORTHIT(REG) ((REG>>2)& 0x1) 
#define SET_FE_DUMMYCTRL_BIT_SHORTHIT(REG) ((1<<2)| REG) 
#define UNSET_FE_DUMMYCTRL_BIT_SHORTHIT(REG) ((~(1<<2)) & REG) 

#define FE_DUMMYCTRL_HITCNT_RANGE_HI		12
#define FE_DUMMYCTRL_HITCNT_RANGE_LOW		3
#define GET_FE_DUMMYCTRL_HITCNT_RANGE(REG) ((REG>>3)&0x1ff) 
#define SET_FE_DUMMYCTRL_HITCNT_RANGE(REG, VAL) ((REG & (~(0x1ff<< 3))) | ((VAL & 0x1ff)<< 3))  


#define FE_DPCTRL_REG	0x11
#define FE_DPCTRL_MASK_RANGE_HI		4
#define FE_DPCTRL_MASK_RANGE_LOW		0
#define GET_FE_DPCTRL_MASK_RANGE(REG) ((REG>>0)&0xf) 
#define SET_FE_DPCTRL_MASK_RANGE(REG, VAL) ((REG & (~(0xf<< 0))) | ((VAL & 0xf)<< 0))  

#define GET_FE_DPCTRL_BIT_PRBSDEC(REG) ((REG>31)& 0x1) 
#define SET_FE_DPCTRL_BIT_PRBSDEC(REG) ((1<<31)| REG) 
#define UNSET_FE_DPCTRL_BIT_PRBSDEC(REG) ((~(1<<31)) & REG) 

#define FE_SUBDET_RESET_REG	0x12
#define GET_FE_SUBDET_REST_BIT_CHIP(REG) ((REG>>0)& 0x1) 
#define SET_FE_SUBDET_REST_BIT_CHIP(REG) ((1<<0)| REG) 
#define UNSET_FE_SUBDET_REST_BIT_CHIP(REG) ((~(1<<0)) & REG) 

#define GET_FE_SUBDET_REST_BIT_DPATH(REG) ((REG>>1)& 0x1) 
#define SET_FE_SUBDET_REST_BIT_DPATH(REG) ((1<<1)| REG) 
#define UNSET_FE_SUBDET_REST_BIT_DPATH(REG) ((~(1<<1)) & REG) 

#define FE_SPICTRL_REGISTER	0x13
#define GET_FE_SPICTRL_BIT_START(REG) ((REG>>5)& 0x1) 
#define SET_FE_SPICTRL_BIT_START(REG) ((1<<5)| REG) 
#define UNSET_FE_SPICTRL_BIT_START(REG) ((~(1<<5)) & REG) 
#define FE_SPICTRL_CHIPID_RANGE_HI		4
#define FE_SPICTRL_CHIPID_RANGE_LOW		0
#define GET_FE_SPICTRL_CHIPID_RANGE(REG) ((REG>>0)&0xf) 
#define SET_FE_SPICTRL_CHIPID_RANGE(REG, VAL) ((REG & (~(0xf<< 0))) | ((VAL & 0xf)<< 0))  

#define FE_SPIDATA_ADDR		0x14
#endif  //#ifndef MUDAQ_REGISTERS__H 
