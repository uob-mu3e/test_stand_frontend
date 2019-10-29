#include "MALIBU.h"

/*
 *
 * LookUp Table for MALIBU v1
Adresses:                                                                                                                                           
GPIOs : GPIO_BP 0111000                                                                                                                             
	(8 bit data:)
	0: ena_AUX1 (PL)                                                                                                                                
	1: ena_AUX2 (PL)                                                                                                                                
	2: SEL_sysCLK  (0:CK_SI0      ; 1:CK_FPGA0)                                                                                                     
	        -> Mainboard: '1'                                                                                                                           
	3: SEL_pllCLK  (0:MCRF connectors ; 1:CK_SI1)                                                                                                   
	        -> Mainboard: '1'                                                                                                                           
	4: SEL_pllTEST (0:PLL_TEST    ; 1:MCRF connectors)                                                                                              
	    -> Mainboard: '0'                                                                                                                           
	5: OE_pllTEST  (1: enabled)                                                                                                                     
	6,7: n.c.                                                                                                                                       
GPIOs : GPIO_1  0111001 ASIC0,1                                                                                                                     
GPIOs : GPIO_2  0111010 ASIC2,3                                                                                                                     
GPIOs : GPIO_3  0111011 ASIC4,5                                                                                                                     
GPIOs : GPIO_4  0111100 ASIC6,7                                                                                                                     
GPIOs : GPIO_5  0111101 ASIC8,9                                                                                                                     
GPIOs : GPIO_6  0111110 ASIC10,11                                                                                                                   
GPIOs : GPIO_7  0111111 ASIC12,13                                                                                                                   
	(8 bit data:)
    0: ASIC0 ena18A (PL)                                                                                                                            
    1: ASIC0 ena18D (PL)                                                                                                                            
    2: ASIC0 SPI_CSn                                                                                                                                
    3: ASIC0 SPI_CSn_CEC (n.c.)                                                                                                                     
    4: ASIC1 ena18A (PL)                                                                                                                            
    5: ASIC1 ena18D (PL)                                                                                                                            
    6: ASIC1 SPI_CSn                                                                                                                                
    7: ASIC1 SPI_CSn_CEC (n.c.)                                                                                                                     
                                                                                                                                                    
MUX1: 1000000                                                                                                                                       
	(4 bit data:)
    0: ASIC1                                                                                                                                        
    1: ASIC2                                                                                                                                        
    2: ASIC3                                                                                                                                        
    3: ASIC0                                                                                                                                        
MUX2: 1000001                                                                                                                                       
	(4 bit data:)
    0: ASIC5                                                                                                                                        
    1: ASIC6                                                                                                                                        
    2: ASIC7                                                                                                                                        
    3: ASIC4                                                                                                                                        
MUX3: 1000010                                                                                                                                       
	(4 bit data:)
    0: ASIC9                                                                                                                                        
    1: ASIC10                                                                                                                                       
    2: ASIC11                                                                                                                                       
    3: ASIC8                                                                                                                                        
MUX4: 1000011                                                                                                                                       
	(4 bit data:)
    0: ASIC13                                                                                                                                       
    1: ASIC14                                                                                                                                       
    2: ASIC15                                                                                                                                       
    3: ASIC12
*/
uint32_t i2c_reg_to_u32(i2c_reg_t i2c_reg){	
	// package the i2c_reg_t data type to alt32; depackage function is in FEB NIOS malibu::u32_to_i2c_reg()
	uint32_t data_u32=0x0; 
	data_u32 += i2c_reg.slave << 16;
	data_u32 += i2c_reg.addr << 8;
	data_u32 += i2c_reg.data;
	return data_u32;
}
bool MALIBU::WriteTo_MALIBU(i2c_reg_t* regs, int n){//
	//n:number of the partern; TODO:need the information from MIDAS //return 1:good; 0:bad
	
	mu.FEB_write((uint32_t) FPGA_ID, data, (uint16_t) n, (uint32_t) START_ADD, (uint32_t) PCIE_MEM_START);
	mu.FEB_write((uint32_t) FPGA_ID, data_arr, (uint16_t) 1, (uint32_t) 0xFFF1, (uint32_t) NEW_PCIE_MEM_START);
	mu.FEB_write((uint32_t) FPGA_ID, data_arr, (uint16_t) 1, (uint32_t) 0xFFF0, (uint32_t) NEW_PCIE_MEM_START);
	usleep(100000);
	return 0;
};
alt_u8 MALIBU::ReadFrom_MALIBU(i2c_reg_t regs){return 0;};//n:number of the partern; TODO:need the information from MIDAS

//==================================================/
bool MALIBU::CheckWrite(i2c_reg_t* regs,int n){
	if(!WriteTo_MALIBU(regs,n)){
		printf("MALIBU::CheckWrite()::Failed to write, Try again...\n");
		if(!WriteTo_MALIBU(regs,n)){
			printf("[ERROR]MALIBU::CheckWrite()::Failed to write!!\n");
			return false;
		}
	}
	return true;
}

//==================================================/
alt_u8 MALIBU::SetPattern(alt_u8 bit_data, alt_u8 bit_mask,bool enable){
	if(enable){
		return  bit_data | bit_mask;		//enable bits in bit_mask and keep other bits same
	}else{
		return	bit_data & ~bit_mask;		//
	}
};

//==================================================/
int MALIBU::Power18A(int asic_ID,bool enable){
	i2c_reg_t reg_curr;
	reg_curr.slave = 0x39 + asic_ID/2;
	reg_curr.addr = 0x01;
	reg_curr.data = 0x00;

	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, A_bit, enable);

	if(!CheckWrite(&reg_curr,1)){
		printf("[ERROR]Failed Power %s 1.8A for asic %d!!\n",enable ? "ON" : "OFF", asic_ID);return -1;
	}
	printf("1.8A of chip %d is %s!!\n",asic_ID,enable ? "ON" : "OFF");return 0;
};

//==================================================/
int MALIBU::Power18D(int asic_ID,bool enable){
	i2c_reg_t reg_curr;
	reg_curr.slave = 0x39 + asic_ID/2;
	reg_curr.addr = 0x01;
	reg_curr.data = 0x00;

	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, D_bit, enable);

	if(!CheckWrite(&reg_curr,1)){
		printf("[ERROR]Failed Power %s 1.8D for asic %d!!\n",enable ? "ON" : "OFF", asic_ID);return -1;
	}
	printf("1.8D of chip %d is %s!!\n",asic_ID,enable ? "ON" : "OFF");return 0;
}

//==================================================/
int MALIBU::EnableSPI(int asic_ID,bool enable){
	i2c_reg_t reg_curr;
	reg_curr.slave = 0x39 + asic_ID/2;
	reg_curr.addr = 0x01;
	reg_curr.data = 0x00;

	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, CS_bit, enable);

	if(!WriteTo_MALIBU(&reg_curr,1)){
		printf("[ERROR]Failed to pll I2C signal line down\n");return -1;
	}
	printf("Ready for chip config\n");return 0;
	
}

//==================================================/
int MALIBU::PowerAUX(bool enable, int i){//i: (0:AUX0, 1:AUX1 )
	i2c_reg_t reg_curr = {0x38,0x01,0x00};
	reg_curr.slave = 0x38;
	reg_curr.addr = 0x01;
	alt_u8 aux_mask;
	if(i==0){
		aux_mask = 0x01;
	}else if(i==1){
		aux_mask = 0x02;
	}else{
		printf("Trying to enable AUX%d: Not exist!!\n",i); return -1;
	}

	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, aux_mask, enable);

	if(!WriteTo_MALIBU(&reg_curr,1)){
		printf("[ERROR]Failed to power %s AUX%d\n",enable ? "ON" : "OFF", i);return -1;
	}
	printf("AUX%d powered %s\n",i, enable ? "ON" : "OFF");return 0;
}


//==================================================/
int MALIBU::sel_sysclk(int opt){//opt:(0:CK_SI0, 1: CK_FPGA0)
	if(opt!=0 and opt!=1){printf("option %d for sysclk selection donnot exist!!\n",opt); return -1;}
	alt_u8 sysclk_mask = 0x04;
	bool enable = (opt==1) ? true : false;
	
	i2c_reg_t reg_curr = {0x38,0x01,0x00};
	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, sysclk_mask, enable);
	
	WriteTo_MALIBU(&reg_curr,1);
	printf("Select sysclk line %d\n", enable ? "CK_FPGA0" : "CK_SI0");
	return 0;
}


//==================================================/
int MALIBU::sel_pllclk(int opt){	//opt:(0: MCRF connector, 1: CK_SI1)	//choose clk line for the pll reference clk
	if(opt!=0 and opt!=1){printf("option %d for sysclk selection donnot exist!!\n",opt); return -1;}
	alt_u8 pllclk_mask = 0x08;
	bool enable = (opt==1) ? true : false;
	
	i2c_reg_t reg_curr = {0x38,0x01,0x00};
	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, pllclk_mask, enable);
	
	WriteTo_MALIBU(&reg_curr,1);
	printf("Select pllclk line %d\n", enable ? "CK_SI1" : "MCRF connector");
	return 0;
	
}

//==================================================/
int MALIBU::sel_pllTest(int opt){	//opt (0: on Board, 1: MCRF connecters) //choose input signal for pll test
	if(opt!=0 and opt!=1){printf("option %d for sysclk selection donnot exist!!\n",opt); return -1;}
	alt_u8 pllTest_mask = 0x10;
	bool enable = (opt==1) ? true : false;
	
	i2c_reg_t reg_curr = {0x38,0x01,0x00};
	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, pllTest_mask, enable);
	
	WriteTo_MALIBU(&reg_curr,1);
	printf("Select pllclk line %d\n", enable ? "CK_SI1" : "MCRF connector");
	return 0;
}

//==================================================/
int MALIBU::Enable_pllTest(bool enable=false){	//enable pll test or not
	alt_u8 en_pllTest_mask = 0x20;
	
	i2c_reg_t reg_curr = {0x38,0x01,0x00};
	reg_curr.data = ReadFrom_MALIBU(&reg_curr);
	reg_curr.data = SetPattern(reg_curr.data, en_pllTest_mask, enable);
	
	WriteTo_MALIBU(&reg_curr,1);
	enable ? printf("Enable PLL test!\n") : printf("Disable PLL test!\n");
	return 0;
}

//==================================================/
int MALIBU::configure_chip(int asic_ID, alt_u8* chip_config_pattern){//send configuration twice /return: (0:good, -1:bad)

	EnableSPI(asic_ID,true);
	//TODO::send configuration bits to Pattern
	EnableSPI(asic_ID,false);
	
};
//==================================================/
int MALIBU::ASIC_Power(int asic_ID, bool enable, const alt_u8* chip_config_pattern){ //Enable 1.8D and 1.8A and configure the chip to ALLOFF
	if(!enable){
		printf("MALIBU::ASIC_Power(): Switching off ASIC %d\n",asic_ID);
		Power18A(asic_ID,enable);
		printf("VCCA18 OFF\n");
		Power18D(asic_ID,enable);
		printf("VCCD18 OFF\n");
		return 0;
	}

	Power18D(asic_ID,enable);
	printf("VCCD18 ON\n");

	if(!configure_chip(asic_ID, chip_config_pattern)){
		printf("ASIC_Power:: configuration fail[chip %d]!\n try again...\n",asic_ID);
		if(!configure_chip(asic_ID, chip_config_pattern)){
			printf("ASIC_Power:: configuration fail[chip %d]!\n ",asic_ID);
			Power18D(asic_ID,enable);
			printf("VCCD18 ON\n");
			return -1;
		}
	}

	Power18A(asic_ID,enable);
	printf("VCCA18 ON\n");
	return 0;
}
//==================================================/
int MALIBU::MALIBU_PowerUp(){
	printf("[malibu] powerup\n");
	WriteTo_MALIBU(malibu_init_regs, sizeof(malibu_init_regs) / sizeof(malibu_init_regs[0]));
	printf("[malibu] powerup DONE\n");
	return 0;//TODO: add some check for the fail of the configuration
}

//==================================================/
int MALIBU::MALIBU_PowerDown(){
	printf("[malibu] powerdown\n");
	WriteTo_MALIBU(malibu_powerdown_regs, sizeof(malibu_powerdown_regs) / sizeof(malibu_powerdown_regs[0]));
	printf("[malibu] powerdown DONE\n");
	return 0;//TODO: add some check for the fail of the configuration
}
//==================================================/
int MALIBU::SetPLLtest(bool enable,bool useCoaxConnectors=false){
	if(!enable){
		Enable_pllTest(enable);
		printf("Disable PLL test");
		return 0;
	}
	int opt = useCoaxConnectors ? 1 : 0;	
	Enable_pllTest(enable);	//enable pll test or not
	sel_pllTest(opt);	//opt (0: on Board, 1: MCRF connecters) //choose input signal for pll test
	printf("Enable PLL test: signal from %d\n",(opt==1)?"MCRF connectors":"on Board");
	return 0;
}
