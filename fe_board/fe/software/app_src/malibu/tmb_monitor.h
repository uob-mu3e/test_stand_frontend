#ifndef TMB_MONITOR_H TMB_MONITOR_H_

#include "../../include/i2c.h"

struct tmb_monitor_t {
    int N_CHIP=1;
    alt_u16 data_all[82];
    int DATA_OFFSET_TMP = 0;
    int DATA_OFFSET_VCC18 = 26;
    int DATA_OFFSET_VCC33 = 78;
    int DATA_OFFSET_SOURCE = 0;
    int DATA_OFFSET_SENSE = 1;
     
    alt_u8 addr_tmp[2] = {0x48,0x49};
    alt_u8 reg_temp_result = 0x00;
    alt_u8 reg_temp_config = 0x01;
    alt_u8 reg_temp_deviceID= 0x0f; //should the readback should be 0x117
    
    alt_u8 addr_VCCD33[2] = {0x4d,0x18};
    alt_u8 addr_MUX[4]={0x40,0x41,0x42,0x43};
    alt_u8 addr_VCC18_monitor = 0x4c;
    //{{{//current monitor related register address and command [PAC1720]
    //=================configuration register========
    //select function register
    alt_u8 reg_sel = 0x00;
    //config command of pac1720
    alt_u8 config_sel_mask[2]={0xfc,0xe7};
    //vsource sampling and averaging
    alt_u8 reg_vsource_config = 0x0a;
    alt_u8 cmd_vsource_config = 0xff; //ch1 and ch2: 20ms sampling time and averaging by 8
    //vsense sampling and averaging
    alt_u8 reg_vsense_sampling[2] = {0x0b,0x0c};
    alt_u8 cmd_vsense_sampling[2] = {0x53,0x53};//80ms;averaging 8; 10mV 
    //alt_u8 cmd_vsense_sampling[2] = {0x5c,0x5c};//80ms;averaging 8; 10mV 
    
    //===================results register=========
    //high limit status register address
    alt_u8 reg_H_limit = {0x04};    
    alt_u8 reg_L_limit = {0x05};    
    //data register of PAC1720 [high byte]
    alt_u8 reg_H_vsource[2]={0x11,0x13};
    alt_u8 reg_H_vsense[2]={0x0d,0x0f};
    //data register of PAC1720 [low byte]
    alt_u8 reg_L_vsource[2]={0x12,0x14};
    alt_u8 reg_L_vsense[2]={0x0e,0x10};
    //}}}
    int mux_index[4] = {3,0,1,2};         // this is the I2C mux fanout 
    
    i2c_t i2c;
    
    alt_u8 I2C_read(alt_u8 slave, alt_u8 addr) {
        alt_u8 data = i2c.get(slave, addr);
        printf("i2c_read: 0x%02X[0x%02X] is 0x%02X\n", slave, addr, data);
        return data;
    }

    alt_u16 I2C_read_16(alt_u8 slave, alt_u8 addr) {
        alt_u16 data = i2c.get16(slave, addr);
        printf("i2c_read: 0x%02X[0x%02X] is 0x%04X\n", slave, addr, data);
        return data;
    }
    
    void I2C_write(alt_u8 slave, alt_u8 addr, alt_u8 data) {
        printf("i2c_write: 0x%02X[0x%02X] <= 0x%02X\n", slave, addr, data);
        i2c.set(slave, addr, data);
    }

    //read constant register
    void read_ProductID(alt_u8 addr);
    void read_ManufID(alt_u8 addr);
    void read_Revision(alt_u8 addr);
    //lower level functions
    void    mux_sel(int id);
    void    ch_sel(alt_u8 slave, int ch); //read sense1 or 2
    void    config_current_monitor(alt_u8 slave); 
    alt_u16  read_vsense(alt_u8 slave, int ch);
    alt_u16 read_vsource(alt_u8 slave,int ch);
    
    //higher level functions
    void    init_current_monitor();
    void    read_VCC33D(int aux); //aux=0 or 1 
    void    read_VCC18(int id);//id=0 to 13;//TODO TMB should be 0 to 12 
    void    read_tmp(int id);//id 0 to 13//TODO test
    
    void    read_tmp_all();
    void    read_power_all();
    void    print_tmp_all();
    void    print_power_all();
    void    read_all(){read_tmp_all(); read_power_all();};
    void    print_all(){print_tmp_all(); print_power_all();};
    //for terminal
    void test_menu();
};

void tmb_monitor_t::read_ProductID(alt_u8 addr){
   printf("ProID[0x%02X]: 0x%02X\n",addr,I2C_read(addr,0xfd)); 
}
void tmb_monitor_t::read_ManufID(alt_u8 addr){
   printf("ManuID[0x%02X]: 0x%02X\n",addr,I2C_read(addr,0xfe)); 
}
void tmb_monitor_t::read_Revision(alt_u8 addr){
   printf("Revision[0x%02X]: 0x%02X\n",addr,I2C_read(addr,0xff)); 
}

void    tmb_monitor_t::ch_sel(alt_u8 slave, int ch){
    I2C_write(slave,reg_sel,I2C_read(slave,reg_sel)&config_sel_mask[ch]);
}
void    tmb_monitor_t::config_current_monitor(alt_u8 slave){
    I2C_write(slave,reg_vsource_config,cmd_vsource_config); //config vsource sampling
    for(int ich=0; ich<2;ich++){
        I2C_write(slave,reg_vsense_sampling[ich],cmd_vsense_sampling[ich]); //config vsense sampling //TODO this has to config to correct value when change the 22mOhm to 4.3 Ohm resistor
    }
}; 

void tmb_monitor_t::mux_sel(int id){
    int mux_id  = id/4;
    int bus     = id%4; 

    for(int i_mux=0; i_mux<4; i_mux++){
        if(i_mux==mux_id){
            I2C_write(addr_MUX[i_mux],0x03,0x80>>mux_index[bus]);
        }else{
            I2C_write(addr_MUX[i_mux],0x03,0x00);
        }
    }
}

void    tmb_monitor_t::init_current_monitor(){
    //init VCC33D monitor
    printf("======INIT CURRENT MONITOR====\n");
    for(int i_aux=0; i_aux<2; i_aux++){
        printf("VCCD33 AUX%d:\n",i_aux);
        read_ProductID(addr_VCCD33[i_aux]);
        read_ManufID(addr_VCCD33[i_aux]);
        read_Revision(addr_VCCD33[i_aux]);
        I2C_write(addr_VCCD33[i_aux],reg_sel,0x1b);//disable v_sensor and v_sense in both channels
        config_current_monitor(addr_VCCD33[i_aux]);
    }
    //init VCC18 monitor
    for(int id=0; id<N_CHIP; id++){
        printf("VCC18 id%d:\n",id);
        mux_sel(id);
        read_ProductID(addr_VCC18_monitor);
        read_ManufID(addr_VCC18_monitor);
        read_Revision(addr_VCC18_monitor);
        I2C_write(addr_VCC18_monitor,reg_sel,0x1b);//disable v_sensor and v_sense in both channels
        config_current_monitor(addr_VCC18_monitor);
    }
    printf("======INIT CURRENT MONITOR DONE====\n");
};


alt_u16  tmb_monitor_t::read_vsense(alt_u8 slave, int ch){//TODO read back the config to decide what range to use
    alt_u8 data_H = I2C_read(slave,reg_H_vsense[ch]);
    alt_u8 data_L = I2C_read(slave,reg_L_vsense[ch]);
    printf("H:0x%02X => %d \n",data_H,data_H);
    printf("L:0x%02X => %d \n",data_L,data_L);
    alt_u16 data = (data_H<<4)+(data_L>>4);// signed data 
    printf("V_sense:0x%04X => %d [*39.06uV]\n",data,data);
    return data;
};
alt_u16 tmb_monitor_t::read_vsource(alt_u8 slave, int ch){
    alt_u8 data_H = I2C_read(slave,reg_H_vsource[ch]);
    alt_u8 data_L = I2C_read(slave,reg_L_vsource[ch]); //TODO check why the lower bits are always 0x00
    printf("H:0x%02X => %d \n",data_H,data_H);
    printf("L:0x%02X => %d \n",data_L,data_L);
    alt_u16 data = (data_H<<3)+(data_L>>5); 
    printf("V_source:0x%04X => %d [*0.019531V]\n",data,data);
    return data;
};

void tmb_monitor_t::read_VCC33D(int aux){
    ch_sel(addr_VCCD33[aux],0);
    data_all[DATA_OFFSET_VCC33+aux*2+DATA_OFFSET_SOURCE] =read_vsource(addr_VCCD33[aux],0);
    data_all[DATA_OFFSET_VCC33+aux*2+DATA_OFFSET_SENSE] =read_vsense(addr_VCCD33[aux],0);
};

void tmb_monitor_t::read_VCC18(int id){
    mux_sel(id);
    //TODO config to 10mV range; resistor set to 4.3 mOhm
    for(int ich=0; ich<2; ich++){ //ch0 is VCC18D
        
        ch_sel(addr_VCC18_monitor,ich);
        I2C_read(addr_VCC18_monitor,reg_vsense_sampling[ich]);//check the configuration
        
        data_all[DATA_OFFSET_VCC18+id*4+ich*2+DATA_OFFSET_SOURCE] =read_vsource(addr_VCC18_monitor,ich);
        data_all[DATA_OFFSET_VCC18+id*4+ich*2+DATA_OFFSET_SENSE]  =read_vsense(addr_VCC18_monitor,ich);
    }
};

void tmb_monitor_t::read_tmp(int id){
    mux_sel(id);
    for(int i_side=0; i_side<2; i_side++){//TODO test, the read back is not what expected
        if( I2C_read_16(addr_tmp[i_side],reg_temp_deviceID)==0x0117){
            data_all[DATA_OFFSET_TMP+id*2+i_side] = I2C_read_16(addr_tmp[i_side],reg_temp_result);
        }else{
            printf("TMP %d [%d]: NOT good!!\n",id,i_side);
        }
    }
};

void tmb_monitor_t::read_tmp_all(){
    for(int id = 0; id<N_CHIP; id++)read_tmp(id);
}

void tmb_monitor_t::read_power_all(){
    read_VCC33D(0);
    read_VCC33D(1);
    for(int id = 0; id<N_CHIP; id++)read_VCC18(id);
}

void tmb_monitor_t::print_tmp_all(){
    for(int id = 0; id<N_CHIP; id++){
        for(int i_side=0; i_side<2; i_side++){
            printf("TMP[%d][%d]:\t 0x%04X\n",id,i_side,data_all[DATA_OFFSET_TMP+id*2+i_side]);
        }
    }
}

void tmb_monitor_t::print_power_all(){
    printf("ID\t V\t V_drop\n");
    for(int aux=0; aux<2; aux++)printf("VCC33D[%d]: 0x%04X\t 0x%04X\n",aux,data_all[DATA_OFFSET_VCC33+aux*2+DATA_OFFSET_SOURCE],data_all[DATA_OFFSET_VCC33+aux*2+DATA_OFFSET_SENSE]);
    for(int id=0; id<N_CHIP; id++){
        for(int ich=0; ich<2; ich++){
            printf(ich==0 ? "VCC18D" : "VCC18A");
            printf("[%d]:\t 0x%04X\t 0x%04X\n",id,data_all[DATA_OFFSET_VCC18+id*4+ich*2+DATA_OFFSET_SOURCE],data_all[DATA_OFFSET_VCC18+id*4+ich*2+DATA_OFFSET_SENSE]);
        }
    }
}
///////======================interface to NIOS terminal============

void tmb_monitor_t::test_menu() {

    while(1) {
        printf("  [0] => monitor VCC33D AUX0\n");
        printf("  [1] => monitor VCC33D AUX1\n");
        printf("  [2] => init current monitor \n");
        printf("  [3] => monitor VCC18 ASIC 0 \n");
        printf("  [4] => monitor temperature 0 \n");
        //printf("  [4] => monitor VCC18D current ASIC 0 \n");
        //printf("  [5] => monitor VCC18A current ASIC 0 \n");
        printf("  [6] => read Product ID \n");
        printf("  [7] => read Manufacturer ID \n");
        printf("  [8] => read Revision \n");
        printf("  [a] => monitor temperature 0:0\n");
        printf("  [b] => monitor temperature 0:1\n");
        printf("  [c] => monitor temperature 1:0\n");
        printf("  [d] => monitor temperature 1:1\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            read_VCC33D(0);
            break;
        case '1':
            read_VCC33D(1);
            break;
        case '2':
            init_current_monitor();
            break;
        case '3':
            read_VCC18(0);
            break;
        case '4':
            read_tmp(0);
            break;
        case '6':
            read_ProductID(addr_VCCD33[1]);
            break;
        case '7':
            read_ManufID(addr_VCCD33[1]);
            break;
        case '8':
            read_Revision(addr_VCCD33[1]);
            break;
        case 'a':
        case 'b':
        case 'c':
        case 'd':
            printf("Not ready yet\n");
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
#endif
