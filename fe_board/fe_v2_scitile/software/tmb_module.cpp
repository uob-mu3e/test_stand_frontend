

#include "tmb_module.h"
#include "tmb_constants.h"
#include "builtin_config/mutrig1_config.h"

//from base.h
#include <stdio.h>
char wait_key(useconds_t us = 100000){return getchar();}


//#include "../../fe/software/sc.h"
//#include "../../../common/include/feb.h"
//#include <altera_avalon_spi.h>


alt_u8 TMB_t::I2C_read(alt_u8 slave, alt_u8 addr) {
    alt_u8 data = i2c.get(slave, addr);
    printf("i2c_read: slave=0x%02X[reg=0x%02X] data is 0x%02X\n", slave, addr, data);
    return data;
}

alt_u16 TMB_t::I2C_read_16(alt_u8 slave, alt_u8 addr) {
    	alt_u16 data = i2c.get16(slave, addr);
    	printf("i2c_read: 0x%02X[0x%02X] is 0x%04X\n", slave, addr, data);
    	data = switch_readout(data);
    	return data;
}

void TMB_t::I2C_write(alt_u8 slave, alt_u8 addr, alt_u8 data) {
    i2c.set(slave, addr, data);
    printf("i2c_write: 0x%02X[0x%02X] <= 0x%02X\n", slave, addr, data);
}

void TMB_t::i2c_write_regs(const i2c_reg_t* regs, int n) {
    for(int i = 0; i < n; i++) {
        auto& reg = regs[i];
        if(reg.slave == 0xFF) {
            usleep(1000);
            continue;
        }
        I2C_write(reg.slave, reg.addr, reg.data);
    }
}

void TMB_t::init_TMB(bool enable){
	//set GPIOs
	for(int id = 0; id <= 3; id++){
		I2C_write(addr_GPIO[id], reg_GPIO_out[0], 0x00);
		I2C_write(addr_GPIO[id], reg_GPIO_out[1], 0x00);
	}
	//set i2c multiplexer GPIOs
	I2C_write(addr_MUX[1], 0x01, 0x00);
	I2C_write(addr_MUX[1], 0x02, 0x1C);

	if(enable){
		for(int id = 0;id <= 3; id++){
			int GPIO_init_id = 2 * (id/3);
			I2C_write(addr_GPIO[id], GPIO_config_reg[0], GPIO_init_values[GPIO_init_id]);
			I2C_write(addr_GPIO[id], GPIO_config_reg[1], GPIO_init_values[GPIO_init_id + 1]);
		}
	}else{
		for(int id = 0;id <=3; id++){
			I2C_write(addr_GPIO[id], GPIO_config_reg[0], 0xff);
			I2C_write(addr_GPIO[id], GPIO_config_reg[1], 0xff);
		}
	}
}

/**
 * Power up/down ASIC
 * UP:
 * - powerup digital 1.8V
 * - configure ALL_OFF pattern twice
 * - if not ok, then powerdown digital and exit
 * - powerup analog 1.8V
 * DOWN:
 * - power down analog 1.8V
 * - power down digital 1.8V
 */
void TMB_t::power_ASIC(int asic, bool enable){
    printf("Powering %s ASIC %d...\n",(enable ? "up" : "down"),asic);
    if(enable){
        //power up
        power_VCC18D(asic,true);
        if(configure_asic(asic, mutrig_config_ALL_OFF) != FEB_REPLY_SUCCESS){
            printf("Configuration error, powering off again\n");
            power_VCC18D(asic,false);
            return;
        }
        power_VCC18A(asic,true);
        printf("[TMB] chip_configure DONE\n");
    }else{
        power_VCC18A(asic,false);
        power_VCC18D(asic,false);
    }
}



//configure gpio lines
//for ASIC power control
//for ASIC CSn lines
void TMB_t::power_VCC18A(int asic, bool enable){
	int GPIO_id = (asic == 12 ? 3 : asic / 3);
	int reg_id = 0;
	int A_bit;
	if(asic <= 8){
		A_bit = VCCA18_first3_index[asic % 3];
		if(A_bit > 9){
			reg_id = 1;
			A_bit -= 10;
		}
	}else{
		A_bit = VCCA18_4_index[asic - 9];
		if(A_bit > 9){
			reg_id = 1;
			A_bit -= 10;
		}
	}

    	if(enable){
        	I2C_write(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id], I2C_read(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id]) | (0x01 << A_bit)); 
    	}else{
        	I2C_write(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id], I2C_read(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id]) & ~(0x01 << A_bit)); 
    	}
       
}

void TMB_t::power_VCC18D(int asic, bool enable){
	int GPIO_id = (asic == 12 ? 3 : asic / 3);
	int reg_id = 0;
	int D_bit;
	if(asic == 0){
		asic = 1;
	}
	if(asic <= 8){
		D_bit = VCCD18_first3_index[asic % 3];
		if(D_bit > 9){
			reg_id = 1;
			D_bit -= 10;
		}
	}else{
		D_bit = VCCD18_4_index[asic - 9];
		if(D_bit > 9){
			reg_id = 1;
			D_bit -= 10;
		}
	}

    	if(enable){
        	I2C_write(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id], I2C_read(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id]) | (0x01 << D_bit)); 
    	}else{
		I2C_write(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id], I2C_read(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id]) & ~(0x01 << D_bit)); 
    	}
     
}

void TMB_t::power_ASIC_all(bool enable){
	if(enable){
		for(int i = 0; i <= 12; i++){
			power_VCC18D(i, true);
			power_VCC18A(i, true);
		}
	}else{
		for(int i = 0; i <= 12; i++){
			power_VCC18A(i, false);
			power_VCC18D(i, false);
		}
	}
}

void TMB_t::setInject(bool enable){
	//set bit GPIO2 of i2c multiplexer at 0x88
        alt_u8 val=I2C_read(addr_MUX[1], 0x01);
	if(enable)
		val |= 1<<4;
	else 
		val &= ~(1<<4);

        I2C_write(addr_MUX[1], 0x01, val); 
}


void TMB_t::SPI_sel(int asic, bool enable){
    	int CS_bit;
	int reg_id = 0;
	int GPIO_id = (asic == 12 ? 3 : asic / 3);
	if(asic <= 2){
		CS_bit = SPI_first3_index[asic % 3];
	}else{
		CS_bit = SPI_first3_index[asic - 9];
	}
	if(CS_bit > 9){
		reg_id = 1;
		CS_bit -= 10;
	}
    	if(enable){
        	I2C_write(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id], I2C_read(addr_GPIO[asic/2], reg_GPIO_out[reg_id]) | (0x01 << CS_bit));
    	}else{
        	I2C_write(addr_GPIO[GPIO_id], reg_GPIO_out[reg_id], I2C_read(addr_GPIO[asic/2], reg_GPIO_out[reg_id]) & ~(0x01 << CS_bit));
    	}
}


void    TMB_t::init_current_monitor(){
    printf("======INIT CURRENT MONITOR====\n");
    for(int id=0; id<N_CHIP+2; id++){
        printf("ID: %d\n",id);
        if(addr_pow_mon[id]==0xff)continue;
        //if(id<N_CHIP)I2C_mux_sel(id);// this is not needed for TMBv1
        check_power_monitor(id,0);
//        read_ProductID(addr_pow_mon[id]);
//        read_ManufID(addr_pow_mon[id]);
//        read_Revision(addr_pow_mon[id]);
        I2C_write(addr_pow_mon[id],0x00,0x00);//00000000: alert is enabled; both channels are enabled; timeout is disabled;
        I2C_write(addr_pow_mon[id],0x01,0x02);//conversion rate: [0x00:1Hz]; [0x01: 2Hz]; [0x02: 4Hz]; [0x03: continuous]
        I2C_write(addr_pow_mon[id],0x03,0x00);// alert: any of the measurement can casuse the alert[source1,source2,sense1,sense2]
        I2C_write(addr_pow_mon[id],0x0a,0xee);//Vsource both//0b11101110 //20ms sampling and 11 bits data' averaging by 4 
        I2C_write(addr_pow_mon[id],0x0b,0x5b);//Vsense0 //0b01011011 //sampling: 80ms; averaging 4; sensor range: -80 to 80mV
        I2C_write(addr_pow_mon[id],0x0c,0x5b);//Vsense1 //0b01011011 //sampling: 80ms; averaging 4; sensor range: -80 to 80mV//TODO should change to different for later 5mOhm resistor
/*
        //set sense limit [the absolute number of voltage drop is based on the configuration]//TODO change this based on later measurement 
        I2C_write(addr_pow_mon[id],0x19,0x7f);//Vsense0 High limit
        I2C_write(addr_pow_mon[id],0x1a,0x7f);//Vsense1 High limit
        I2C_write(addr_pow_mon[id],0x1b,0x80);//Vsense0 Low limit
        I2C_write(addr_pow_mon[id],0x1c,0x80);//Vsense1 Low limit
        if(id<N_CHIP){//set vsource limit for 1.8V
            I2C_write(addr_pow_mon[id],0x1d,0x0d);//Vsource0 High limit 0x18:2.5V
            I2C_write(addr_pow_mon[id],0x1e,0x0d);//Vsource1 High limit 0x18:2.5V
            I2C_write(addr_pow_mon[id],0x1f,0x0a);//Vsource0 Low  limit 0x14:1.565V
            I2C_write(addr_pow_mon[id],0x20,0x0a);//Vsource1 Low  limit 0x14:1.565V
        
        }else{//set vsource limit for 1.8V
            I2C_write(addr_pow_mon[id],0x1d,0x18);//Vsource0 High limit 0x18:3.75V
            I2C_write(addr_pow_mon[id],0x1e,0x18);//Vsource1 High limit 0x18:3.75V
            I2C_write(addr_pow_mon[id],0x1f,0x14);//Vsource0 Low  limit 0x14:3.125V
            I2C_write(addr_pow_mon[id],0x20,0x14);//Vsource1 Low  limit 0x14:3.125V
        }
*/    
        printf("ID: %d done\n",id);
    }
    printf("======INIT CURRENT MONITOR DONE====\n");
}

void TMB_t::I2C_mux_sel(int gid){
    int id=I2C_bus_index[id];
    int mux_id  = id/4;
    int bus     = id%4; 
    printf("mux: id=%d ->[muxid=%d,bus=%d]\n",id,mux_id,bus);
    for(int i_mux=0; i_mux<4; i_mux++){
        I2C_write(addr_MUX[i_mux],0x03,(i_mux == mux_id ? 0x80>>I2C_mux_index[bus] : 0x00));
    }
}

void TMB_t::I2C_bus_sel(int id){
	I2C_write(0x5d, 0x03, 0x00);
	alt_u8 data = 0x00;
	alt_u8 reg = 0x03;
	if(id <= 6 && id >= 0){
		data |= 1UL << (8 - I2C_bus_index[id]); //bitwise OR function to activate to bus corresponding bit.
	}
	alt_u8 mux_addr = addr_MUX[id/4];
	if(id > 6){
		printf("bus id too large");
		return;
	}
	I2C_write(mux_addr, reg, data);
}

alt_u8 TMB_t::get_temperature_address(int id){
	int bus_id,id_addr;
	alt_u8 addr;
	if(id > 27 || id < 0){
		printf("invalid tempreture sensor id");
		return 0;
	}
	if(id >= 0 && id <= 23){
		id_addr = (id/2)%4;   //opposing matrices have the same adress. They reapeat every 4 ASICs
		bus_id = 2*(id/8)+(id%2);	//due to location of the subbusses on the board and the ids of the matrices this formula yields the subbus for each matrix
		addr = addr_tmp[id_addr];
	}else{
		addr = I2C_tmp_addr_over23[id-24];
		bus_id = I2C_tmp_bus_over23[id-24];
	}
	I2C_bus_sel(bus_id);
	return addr;
}

alt_u16  TMB_t::read_vsense(int id, int ch){ //id gives chip id, ch gives analog (0) or digital (1)
    	if(id >= 0 && id <= 11){
		int addr_id = (id/2) % 2;
		int bus_id = 2 * (id/4) + ch;

		I2C_bus_sel(bus_id);	
		alt_u16 vsense = I2C_read_16(addr_pow_mon[addr_id],reg_vsense[id%2]);
		vsense /= 0x0010;
   		printf("V_sense:0x%04X => %d [*39.06uV]\n",vsense,vsense);
    		return vsense;
	}else if(id == 13){
		I2C_bus_sel(0);
		alt_u16 vsense = I2C_read_16(addr_pow_mon[2], reg_vsense[ch]);
		vsense = ((~(vsense >> 4)) & 0x0FFF);
		printf("V_sense:0x%04X => %d [*39.06uV]\n",vsense,vsense);
		return vsense;
	}else{
		return 0x0000;
	}
}
alt_u16 TMB_t::read_vsource(int id, int ch){ //id gives chip id, ch gives analog (0) or digital (1)
	if(id >= 0 && id <= 11){
		int addr_id = (id/2) % 2;
		int bus_id = 2 * (id/4) + ch;

		I2C_bus_sel(bus_id);
		alt_u16 vsource = I2C_read_16(addr_pow_mon[addr_id], reg_vsource[id%2]);
		vsource /= 0x0020;
		printf("Temperature for id %d, ch %d is: %04X", id, ch, vsource);
		return vsource;
    	}else if(id == 13){
		I2C_bus_sel(0);
		alt_u16 vsource = I2C_read_16(addr_pow_mon[2], reg_vsource[ch]);
		vsource /= 0x0020;
		printf("Temperature for id %d, ch %d is: %04X", id, ch, vsource);
		return vsource;
	}else{
		return 0x0000;
	}
}

void TMB_t::read_pow_limit(int id){
    data_all_powerStat[id] = ((I2C_read(addr_pow_mon[id],reg_HL))<<4)+((I2C_read(addr_pow_mon[id],reg_LL))&0x0f);
}

alt_u16 TMB_t::read_temperature_sensor(int id){
	alt_u8 addr = get_temperature_address(id);
	data_all_tmp[id] = I2C_read_16(addr,reg_temp_result);
	if(id < 26){
 	   	printf("TMP %d: 0x%04X => %d [*7.8125C]!!\n",id,data_all_tmp[id],data_all_tmp[id]);
	}else{
		printf("TMP %d: 0x %04X => %d [*0.0078125C]!!\n",id,data_all_tmp[id],data_all_tmp[id]);
	}
	return data_all_tmp[id];
}

alt_u16 TMB_t::switch_readout(alt_u16 tmp){
	alt_u16 ordered, back_part;
	alt_u8 front_part;

	front_part = tmp;
	back_part = tmp - front_part;
	ordered = (back_part / 0x0100) + (front_part * 0x0100);
	return ordered;
}

bool TMB_t::check_power_monitor(int id, int ch){
	if(id >= 0 && id <= 11){
		int addr_id = (id/2)%2;
		int bus_id = 2 * (id/4) + ch;

		I2C_bus_sel(bus_id);
		int addr = addr_pow_mon[addr_id];
		return (I2C_read(addr, 0xfd) == 0x57 ? true : false);
	}if(id == 13){
		I2C_bus_sel(0);
		return (I2C_read(addr_pow_mon[2], 0xfd) == 0x57 ? true : false);
	}else{
		return false;
	} 
}

void TMB_t::check_power_monitor_all(){
	bool power_status[28];
	for(int id = 0; id <= 13; id++){
		power_status[2*id] = check_power_monitor(id,0);
		power_status[2*id + 1] = check_power_monitor(id,1);
	}
	printf("   0  1  2  3  4  5  6  7  8  9  10 11 12 3V\nA  ");
	for(int id = 0; id <= 13; id++){
		printf("%d  ", power_status[2*id]);
	}
	printf("\nD  ");
	for(int id = 0; id <= 13; id++){
		printf("%d  ", power_status[2*id + 1]);
	}
}

bool TMB_t::check_temperature_sensor(int id){
	alt_u8 addr = get_temperature_address(id);
	if(id <= 25){
		if(I2C_read_16(addr,reg_temp_deviceID[0]) == temp_deviceID_result[0]){
        		return  true ;
    		}else{
        		return false ;
    		}
	}else{
		if(I2C_read_16(addr,reg_temp_deviceID[1]) == temp_deviceID_result[1]){
			return true;
		}else{
			return false;
		}
	}
    
}

void TMB_t::check_temperature_sensor_all(){
	bool tmp_status[28];
	for(int i = 0; i < 28; i++){
		tmp_status[i] = check_temperature_sensor(i);
	}
	printf("0  1  2  3  4  5  6  7  8  9  10 11 12 TMB\n");
	for(int i = 0; i < 14; i++){
		printf("%d  ",tmp_status[2*i]);
	}
	printf("\n");
	for(int i = 0; i < 14; i++){
		printf("%d  ", tmp_status[2*i +1]);
	}
	printf("\n");
}

void TMB_t::read_tmp_all(){
    	for(int id = 0; id<28; id++){
        	read_temperature_sensor(id);
    	}
}

void TMB_t::read_power_all(){
    	for(int id = 0; id<14; id++){
        	for(int ch=0; ch<2; ch++){
			printf("ID,A/D: [%d,%d]\n",id,ch);
            		data_all_power[id*4+ch*2]=read_vsource(id, ch);
            		data_all_power[id*4+ch*2+1]=read_vsense(id,ch);
        	}
    	}
}

void TMB_t::print_tmp_all(){
    for(int id = 0; id<28; id++){
            printf("TMP[%d]:\t 0x%04X\n",id,data_all_tmp[id]);        
    }
}

void TMB_t::print_power_all(){
    	printf("ID\t \t V\t V_drop\n");
    	for(int id=0; id<=13; id++){
        	for(int ich=0; ich<2; ich++){
            		if(id <=12){
                		printf(ich==0 ? "VCC18A" : "VCC18D");
            		}else{
                		printf(ich==0 ? "VCC33" : "VCC36");
            		}
			if(id == 12){
				printf("[12]:\t no monitor implemented\n");
			}else if(id == 0 && ich == 1){
				printf("[%d]:\t 0x%04X\t 0x%04X\n",id,data_all_power[6],data_all_power[7]);
			}else{
            			printf("[%d]:\t 0x%04X\t 0x%04X\n",id,data_all_power[id*4+ich*2],data_all_power[id*4+ich*2+1]);
			}
		}
    	}
}


//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int TMB_t::spi_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern) {
	int status=0;
/* FEB
	uint16_t rx_pre=0xff00;
//        printf("tx | rx\n");
	uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
       	do{
		nb--;
		//do spi transaction, one byte at a time
                alt_u8 rx = 0xCC;
                alt_u8 tx = bitpattern[nb];

                alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
                rx = IORD_8DIRECT(SPI_BASE, 0);
//                printf("%02X %02x\n",tx,rx);
//                printf("%02X ",tx);

		//pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
		unsigned char rx_check= (rx_pre | rx ) >> (8-MUTRIG_CONFIG_LEN_BITS%8);
		if(nb==MUTRIG_CONFIG_LEN_BYTES-1){
			rx_check &= 0xff>>(8-MUTRIG_CONFIG_LEN_BITS%8);
		};

		if(rx_check!=bitpattern[nb]){
//			printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
			status=-1;
		}
		rx_pre=rx<<8;
	}while(nb>0);
//        printf("\n");
*/
	return status;
}



void TMB_t::print_config(const alt_u8* bitpattern) {
	uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
	do{
		nb--;
                printf("%02X ",bitpattern[nb]);
	}while(nb>0);
}


//configure ASIC
alt_u16 TMB_t::configure_asic(alt_u32 asic, const alt_u8* bitpattern) {
    printf("[TMB] chip_configure(%u)\n", asic);

    int ret;
    SPI_sel(asic,true);
    ret = spi_write_pattern(0, bitpattern);
    SPI_sel(asic,false);
    usleep(0);
    SPI_sel(asic,true);
    ret = spi_write_pattern(0, bitpattern);
    SPI_sel(asic,false);

    if(ret != 0) {
//        printf("[scifi] Configuration error\n");
        return FEB_REPLY_ERROR;
    }

    return FEB_REPLY_SUCCESS;
}


//FEB using namespace mu3e::daq::feb;
//TODO: add list&document in specbook
//TODO: update functions
//Callback function called after receiving a command from the FEB slow control interface
alt_u16 TMB_t::sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
/* FEB
    switch(cmd) {
    case CMD_TILE_TMB_ON:
        power_TMB(true);
        break;
    case CMD_TILE_TMB_OFF:
        power_TMB(false);
        break;

    case CMD_TILE_ON:
        power_TMB(true);
	//TODO: automatic ASIC powering
        break;
    case CMD_TILE_OFF:
        power_TMB(false);
	//TODO: automatic ASIC powering
	//TODO: test if this kind of slow powering down scheme is needed
        break;
	alt_u16 test[4] = {0xf50f,0xf68f,0xf65f,0xf51f};
	alt_u16 out;
	for(int i = 0; i<4; i++){	
		out = test[i] / 0x0020;
		printf("\n %04X	\n", out);
	}
	return 0;
    case CMD_TILE_TEMPERATURES_READ:
	data_all_tmp=(alt_u16*)data;
	read_tmp_all();
        break;
    case CMD_TILE_POWERMONITORS_READ:
	data_all_power=(alt_u16*)data;
	read_power_all();
        break;
    default:
        int asic = cmd & 0x000F;
    	switch(cmd & 0xFFF0){
            case CMD_MUTRIG_ASIC_CFG:
                printf("configuring ASIC\n");
                configure_asic(asic, (alt_u8*)data);
                break;
            case CMD_TILE_ASIC_ON:
                power_ASIC(asic,true);
	alt_u16 test[4] = {0xf50f,0xf68f,0xf65f,0xf51f};
	alt_u16 out;
	for(int i = 0; i<4; i++){	
		out = test[i] / 0x0020;
		printf("\n %04X	\n", out);
	}
	return 0;
                break;
            case CMD_TILE_ASIC_OFF:
                power_ASIC(asic,false);
                break;
            default:
                printf("[sc_callback] unknown command\n");
                break;
            }
    }
*/
        return 0;
}
void TMB_t::menu_TMB_main(){
//FEB    auto& regs = sc.ram->regs.TMB;

    while(1) {
//        printf("  [0] => reset\n");
        printf("  [1] => powerup MALIBU\n");
        printf("  [2] => powerdown MALIBU\n");
        printf("  [3] => ASIC power control\n");
        printf("  [4] => datapath status\n");
        printf("  [5] => tmb debug menu\n");
        printf("  [6] => monitor menu\n");
	printf("  [7] => ---\n");
	printf("  [8] => powerdown all ASICs\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            init_TMB(true);
            break;
        case '2':
            init_TMB(false);
            break;
        case '3':
	    menu_TMB_ASIC();
            //power_ASIC(0);
            break;
        case '4':
            printf("// not implemented\n");
//FEB            printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X\n", regs.mon.status);
//FEB            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '5':
            menu_TMB_debug();
            break;
        case '6':
            menu_TMB_monitors();
            break;
	case '7':
	    //power_ASIC_all(true);
	    break;
	case '8':
	    power_ASIC_all(false);
	    break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void TMB_t::menu_TMB_debug() {
    alt_u8 rx = 0xCC;
    alt_u8 tx = 0xAA;

    while(1) {
        printf("  [0] => check power monitors\n");
        printf("  [1] => check temperature sensors\n");
        printf("  [2] => try all I2C addresses\n");
        printf("  [3] => try SPI 32b transaction\n");
        printf("  [i] => disable pulse injection\n");
        printf("  [I] => enable pulse injection\n");

        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            check_power_monitor_all();
            break;
        case '1':
            check_temperature_sensor_all();
            break;
        case '2':
            for(int i=0x10;i<0x7f;i++){
                    auto ret=I2C_read(i,0);
                    printf("@%2.2x: %2.2x\n",i,ret);
            }
            break;
        case '3':
            printf("// not implemented\n");
            //  alt_avalon_spi_command(SPI_BASE, 0, 1, &tx, 0, &rx, 0);
            break;
        case 'I':
	    setInject(true);
	    break;
        case 'i':
	    setInject(false);
	    break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void TMB_t::menu_TMB_monitors() {

    while(1) {
        printf("  [0] => read power\n");
        printf("  [1] => read temperature\n");
        printf("  [2] => debug\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            read_power_all();
            print_power_all();
            break;
        case '1':
            read_tmp_all();
            print_tmp_all();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

void TMB_t::menu_TMB_ASIC(){
	
	int i=0;
	int a_d=0;
	while(1){
		printf("CURRENTLY SELECTED ASIC: %d%c\n", i, (a_d == 0 ? 'a' : 'd'));
		printf("  [0] => turn off ASIC\n");
		printf("  [1] => turn on ASIC\n");
		printf("  [+] => increase ASIC id\n");
	       	printf("  [-] => decrease ASIC id\n");
		printf("  [s] => go to specific ASIC\n");
		printf("  [d] => digital\n");
		printf("  [a] => analog\n");
		printf("  [e] => turn off all ASICs\n");
		printf("  [q] => exit\n");

		printf("Select entry ...\n");
		char cmd = wait_key();
		switch(cmd){
			case '0':
				if(a_d == 0){
					power_VCC18A(i,false);
				}else if(a_d == 1){
					power_VCC18D(i,false);
				}
				break;
			case '1':
				if(a_d == 1){
					power_VCC18D(i,true);
				}else if(a_d == 0){
					power_VCC18A(i,true);
				}
				break;
			case '+':
				i = (i + 1) % 13;
				break;
			case '-':
				i = (i - 1) % 13;
				break;
			case 's':
				while(1){
					i = wait_key() % 13;
					break;
				}
				break;
			case 'a':
				a_d = 0;
				break;
			case 'd':
				a_d = 1;
				break;
			case 'e':
				power_ASIC_all(false);
				break;
			case 'q':
				return;
		}
	}
}
