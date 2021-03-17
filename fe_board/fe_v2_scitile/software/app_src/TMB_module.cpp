

#include "TMB_module.h"
#include "TMB_constants.h"



// send u32 to i2c
void malibu_t::i2c_write_u32(volatile alt_u32* data_u32, int n) {
    for(int i = 0; i < n; i++) {
        i2c_reg_t reg = u32_to_i2c_reg(data_u32[i]);
        i2c_write_regs(&reg, 1);
    }
}

// save the data to i2c_reg_t struct
i2c_reg_t malibu_t::u32_to_i2c_reg(alt_u32 data_u32) {
// TODO: KB - not working like this, for what do we need it?
/*
    i2c_reg_t i2c_reg = {
        data_u32 & 0x00FF0000,
        data_u32 & 0x0000FF00,
        data_u32 & 0x000000FF
    };
*/
    i2c_reg_t i2c_reg = {
        alt_u8(data_u32 >> 16),
        alt_u8(data_u32 >> 8),
        alt_u8(data_u32 >> 0),
    };    return i2c_reg;
}

alt_u8 malibu_t::I2C_read(alt_u8 slave, alt_u8 addr) {
    alt_u8 data = i2c.get(slave, addr);
    printf("i2c_read: 0x%02X[0x%02X] is 0x%02X\n", slave, addr, data);
    return data;
}

alt_u16 malibu_t::I2C_read_16(alt_u8 slave, alt_u8 addr) {
    alt_u16 data = i2c.get16(slave, addr);
    printf("i2c_read: 0x%02X[0x%02X] is 0x%04X\n", slave, addr, data);
    return data;
}

void malibu_t::I2C_write(alt_u8 slave, alt_u8 addr, alt_u8 data) {
    printf("i2c_write: 0x%02X[0x%02X] <= 0x%02X\n", slave, addr, data);
    i2c.set(slave, addr, data);
}

alt_u8 malibu_t::spi_write(alt_u32 slave, alt_u8 w) {
    alt_u8 r = 0xCC;
    //        printf("spi_write: 0x%02X\n", w);
    alt_avalon_spi_command(SPI_BASE, slave, 1, &w, 0, &r, 0);
    r = IORD_8DIRECT(SPI_BASE, 0);
    //        printf("spi_read: 0x%02X\n", r);
    return r;
}

void malibu_t::i2c_write_regs(const i2c_reg_t* regs, int n) {
    for(int i = 0; i < n; i++) {
        auto& reg = regs[i];
        if(reg.slave == 0xFF) {
            usleep(1000);
            continue;
        }
        I2C_write(reg.slave, reg.addr, reg.data);
    }
}

void malibu_t::power_TMB(bool enable) {
    printf("power %s TMB\n",(enable ? "up" : "down"));
    if(enable){
        i2c_write_regs(malibu_init_regs, sizeof(malibu_init_regs) / sizeof(malibu_init_regs[0]));
        init_current_monitor();
    }else{
        i2c_write_regs(malibu_powerdown_regs, sizeof(malibu_powerdown_regs) / sizeof(malibu_powerdown_regs[0]));
    }
    printf("power %s TMB DONE\n",(enable ? "up" : "down"));
}

void malibu_t::power_ASIC(int asic, bool enable){
    printf("Powering %s ASIC %d...\n",(enable ? "up" : "down"),asic);
    if(enable){
        chip_configure(asic, stic3_config_ALL_OFF);
    }else{
        power_VCC18A(asic,false);
        power_VCC18D(asic,false);
    }
}

void malibu_t::power_VCC18A(int asic, bool enable){
    A_bit = 1 << (0 + asic%2*4);//TODO check this for TMB, current algorithm is based on malibu
    if(enable){
        I2C_write(addr_GPIO[asic/2], 0x01, I2C_read(addr_GPIO[asic/2], 0x01) | A_bit); 
    }else{
        I2C_write(addr_GPIO[asic/2], 0x01, I2C_read(addr_GPIO[asic/2], 0x01) & ~A_bit); 
    }
       
}

void malibu_t::power_VCC18D(int asic, bool enable){

    D_bit = 1 << (1 + asic%2*4);//TODO check this for TMB, current algorithm is based on malibu
    if(enable){
        I2C_write(addr_GPIO[asic/2], 0x01, I2C_read(addr_GPIO[asic/2], 0x01) | D_bit);
    }else{
        I2C_write(addr_GPIO[asic/2], 0x01, I2C_read(addr_GPIO[asic/2], 0x01) & ~D_bit);
    }
}

void malibu_t::SPI_sel(int asic, bool enable){
    CS_bit = 1 << (2 + asic%2*4);//TODO check this for TMB, current algorithm is based on malibu
    if(enable){
        I2C_write(addr_GPIO[asic/2], 0x01, I2C_read(addr_GPIO[asic/2], 0x01) & ~CS_bit);
    }else{
        I2C_write(addr_GPIO[asic/2], 0x01, I2C_read(addr_GPIO[asic/2], 0x01) | CS_bit);
    }
}


void    malibu_t::init_current_monitor(){
    printf("======INIT CURRENT MONITOR====\n");
    for(int id=0; id<N_CHIP+2; id++){
        printf("ID: %d\n",id);
        if(addr_pow_mon[id]==0xff)continue;
        if(id<N_CHIP)I2C_mux_sel(id);// this is not needed for TMBv1
        read_ProductID(addr_pow_mon[id]);
        read_ManufID(addr_pow_mon[id]);
        read_Revision(addr_pow_mon[id]);
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

void malibu_t::I2C_mux_sel(int id){
    int mux_id  = id/4;
    int bus     = id%4; 
    printf("%d [%d,%d]\n",id,mux_id,bus);
    for(int i_mux=0; i_mux<4; i_mux++){
        I2C_write(addr_MUX[i_mux],0x03,(i_mux == mux_id ? 0x80>>I2C_mux_index[bus] : 0x00));
    }
}

alt_u16  malibu_t::read_vsense(int id, int ch){
    if(addr_pow_mon[id]==0xff)return 0x0000;
    alt_u16 data = (I2C_read_16(addr_pow_mon[id],reg_vsense[ch]))>>4;
    printf("V_sense:0x%04X => %d [*39.06uV]\n",data,data);
    return data;
}
alt_u16 malibu_t::read_vsource(int id, int ch){
    printf("0x%04X 0x%04X\n",addr_pow_mon[id],reg_vsource[ch]);
    if(addr_pow_mon[id]==0xff)return 0x0000;
    alt_u16 data = I2C_read_16(addr_pow_mon[id],reg_vsource[ch])>>5; 
    printf("V_source:0x%04X => %d [*0.019531V]\n",data,data);
    return data;
}

void malibu_t::read_pow_limit(int id){
    data_all_powerStat[id] = ((I2C_read(addr_pow_mon[id],reg_HL))<<4)+((I2C_read(addr_pow_mon[id],reg_LL))&0x0f);
}

void malibu_t::read_tmp(int chip_id){
    //I2C_mux_sel(chip_id); //TODO need this for multiple readout
    for(int i_side=0; i_side<2; i_side++){
        if(read_tmp_deviceID(chip_id,i_side)){ 
            data_all_tmp[chip_id*2+i_side] = I2C_read_16(addr_tmp[i_side],reg_temp_result);
            printf("TMP %d [%d]: 0x%04X => %d [*7.8125C]!!\n",chip_id,i_side,data_all_tmp[chip_id*2+i_side],data_all_tmp[chip_id*2+i_side]);
        }else{
            printf("TMP %d [%d]: NOT good!!\n",chip_id,i_side);
        }
    }
}

bool malibu_t::read_tmp_deviceID(int id, int i_side){
    if ( I2C_read_16(addr_tmp[i_side],reg_temp_deviceID)){
        return  true ;
    }else{
        return false ;
    }
}

void malibu_t::read_tmp_all(){
    //for(int id = 0; id<N_CHIP; id++)read_tmp(id);
    read_tmp(0);//MALIBU
}

void malibu_t::read_power_all(){
    for(int id = 0; id<N_CHIP+2; id++){
        for(int ch=0; ch<2; ch++){
            if(id>=N_CHIP and ch==1)continue;
            printf("ID,side: [%d,%d]\n",id,ch);
            data_all_power[id*4+ch*2]=read_vsource(id, ch);
            data_all_power[id*4+ch*2+1]=read_vsense(id,ch);
        }
    }
}

void malibu_t::print_tmp_all(){
    for(int id = 0; id<N_CHIP; id++){
        for(int i_side=0; i_side<2; i_side++){
            printf("TMP[%d][%d]:\t 0x%04X\n",id,i_side,data_all_tmp[id*2+i_side]);
        }
    }
}

void malibu_t::print_power_all(){
    printf("ID\t V\t V_drop\n");
    for(int id=0; id<N_CHIP+2; id++){
        for(int ich=0; ich<2; ich++){
            if(ich<N_CHIP){
                printf(ich==0 ? "VCC18D" : "VCC18A");
            }else{
                printf("VCC33:");
            }
            printf("[%d]:\t 0x%04X\t 0x%04X\n",id,data_all_power[id*4+ich*2],data_all_power[id*4+ich*2+1]);
        }
    }
}


//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SPI_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern) {
    int status=0;
    uint16_t rx_pre=0xff00;
    for(int nb=STIC3_CONFIG_LEN_BYTES-1; nb>=0; nb--){
        alt_u8 rx = malibu_t::spi_write(spi_slave, bitpattern[nb]);
        //pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
        unsigned char rx_check= (rx_pre | rx ) >> (8-STIC3_CONFIG_LEN_BITS%8);
        if(nb==STIC3_CONFIG_LEN_BYTES-1){
            rx_check &= 0xff>>(8-STIC3_CONFIG_LEN_BITS%8);
        };

        if(rx_check!=bitpattern[nb]){
//            printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
            status=-1;
        }
        rx_pre=rx<<8;
    }
    return status;
}

//configure a specific ASIC returns 0 if configuration is correct, -1 otherwise.
int SPI_configure(alt_u32 slaveAddr, const unsigned char* bitpattern) {
    //configure SPI. Note: pattern is not in full bytes, so validation gets a bit more complicated. Shifting out all bytes, and need to realign after.
    //This is to be done still
    int ret;
    ret=SPI_write_pattern(slaveAddr, bitpattern);
    ret=SPI_write_pattern(slaveAddr, bitpattern);

    return ret;
}

/**
 * Configure ASIC
 *
 * - powerup digital 1.8V
 * - configure pattern
 * - configure and validate
 * - if not ok, then powerdown digital and exit
 * - powerup analog 1.8V
 */
int malibu_t::chip_configure(int asic, const alt_u8* bitpattern) {
    printf("[malibu] chip_configure(%u)\n", asic);

    alt_u8 spi_slave = 1;

    power_VCC18D(asic,true);
    int ret;
    SPI_sel(asic,true);
    ret = SPI_write_pattern(spi_slave, bitpattern);
    SPI_sel(asic,false);

    SPI_sel(asic,true);
    ret = SPI_write_pattern(spi_slave, bitpattern);
    SPI_sel(asic,false);
    /*
    if(ret != 0) {
        printf("Configuration error, powering off again\n");
        power_VCC18D(asic,false);
        return -1;
    }
    */
    //TODO this has to be un-comment 
    power_VCC18A(asic,true);
    printf("[malibu] chip_configure DONE\n");
    return 0;
}

//#include "../../../../common/include/feb.h"
//using namespace mu3e::daq::feb;
//TODO: add list&document in specbook
//TODO: update functions
alt_u16 malibu_t::sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    switch(cmd) {
    case CMD_TILE_ON:
        power_TMB(true);
        break;
    case CMD_TILE_OFF:
        power_TMB(false);
        break;
    case CMD_TILE_STIC_OFF:
        chip_configure(0, stic3_config_ALL_OFF);
        break;
    case CMD_TILE_STIC_PLL_TEST:
        chip_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
        break;
    case CMD_TILE_I2C_WRITE:
        i2c_write_u32(data, n);
        break;
    default:
        if((cmd & 0xFFF0) == CMD_TILE_STIC_CFG) {
            printf("try stic_configure\n");
            int asic = cmd & 0x000F;
            chip_configure(asic, (alt_u8*)data);
        }
        else {
            printf("[sc_callback] unknown command\n");
        }
    }

    return 0;
}

void malibu_t::menu_TMB_main() {
    auto& regs = sc.ram->regs.malibu;

    while(1) {
        printf("  [0] => reset\n");
        printf("  [1] => powerup MALIBU\n");
        printf("  [2] => powerdown MALIBU\n");
        printf("  [3] => powerup ASIC 0\n");
        printf("  [4] => stic3_config_PLL_TEST_ch0to6_noGenIDLE\n");
        printf("  [5] => data\n");
        printf("  [6] => monitor test\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            power_TMB(true);
            break;
        case '2':
            power_TMB(false);
            break;
        case '3':
            power_ASIC(0);
            break;
        case '4':
            chip_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
            break;
        case '5':
            printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X\n", regs.mon.status);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '6':
            menu_TMB_monitors();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

void malibu_t::menu_TMB_monitors() {

    while(1) {
        printf("  [0] => read power\n");
        printf("  [1] => read temperature\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            read_power_all();
            break;
        case '1':
            read_tmp_all();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
