#include "mutrig_device.h"

#include<cstring>
#include<stdexcept>
#include<string>
#include<chrono>
#include<thread>
#include<algorithm>
#include<iterator>

namespace mudaq {

MutrigDevice::MutrigDevice(const std::string& path) :
  DmaMudaqDevice(path)
  {
    mem_owner = NONE;
    _pagesize(); // to avoid compiler warning 
  }

/**
 * Claims memory interphase for slowcontrol
 * data path gets disabled
 */
void MutrigDevice::claimMem(owner own) {
    if(isMemNot(NONE)) throw std::runtime_error("Mem interface is not in NONE control state. Can not be claimed.");
    if(own == SLOW) {
        uint32_t cmd_reg;
        cmd_reg = read_register_rw(MUTRIG_CMD_REGISTER_W);
        write_register(MUTRIG_CMD_REGISTER_W, SET_MUTRIG_CMD_BIT_SLOW_ENABLE(cmd_reg)); // send config to FPGA
    } //else configInMem = uint32_t(-1);
    mem_owner = own;
}

/**
 * Releases memory interphase for slowcontrol
 */
void MutrigDevice::releaseMem() {
    //if(mem_owner == SLOW) { // default state on FPGA is data
        uint32_t cmd_reg;
        cmd_reg = read_register_rw(MUTRIG_CMD_REGISTER_W);
        write_register(MUTRIG_CMD_REGISTER_W, UNSET_MUTRIG_CMD_BIT_SLOW_ENABLE(cmd_reg)); // send config to FPGA
    //}
    mem_owner = NONE;
    //configInMem = uint32_t(-1);
}

/**
 * Checks if mem is in owner control state (owner and registers)
 */
int MutrigDevice::isMemNot(owner own) const {
    if(mem_owner != own) return 1; //throw std::runtime_error("Mem interface is not in slow control state. Owned by id "+std::to_string(mem_owner));
    uint32_t cmd_reg;
    cmd_reg = read_register_rw(MUTRIG_CMD_REGISTER_W);
    if(own == SLOW) {
        if(GET_MUTRIG_CMD_BIT_SLOW_ENABLE(cmd_reg) != 1) return 2; //throw std::runtime_error("Mem interface is not in slow control state (according to registers)");
    }
    else {
        if(GET_MUTRIG_CMD_BIT_SLOW_ENABLE(cmd_reg) == 1) return 2;
    }
    return 0; 
}

void MutrigDevice::setDummyConfig(bool dummy) {
    uint32_t cmd_reg;
    cmd_reg = read_register_rw(MUTRIG_CMD_REGISTER_W);
    if(dummy) cmd_reg =   SET_MUTRIG_CMD_BIT_DUMMY_CONFIG_ENABLE(cmd_reg);
    else      cmd_reg = UNSET_MUTRIG_CMD_BIT_DUMMY_CONFIG_ENABLE(cmd_reg);
    write_register(MUTRIG_CMD_REGISTER_W, cmd_reg); // reset config 
}


void MutrigDevice::setDummyData(bool dummy, int n_, bool fast) {
    uint32_t cmd_reg;
    cmd_reg = read_register_rw(MUTRIG_DPATH_REGISTER_W);
    // enable dummy data MUTRIG_DPATH_BIT_DUMMY_DATA_ENABL
    if(dummy) cmd_reg =   SET_MUTRIG_DPATH_BIT_DUMMY_DATA_ENABLE(cmd_reg);
    else      cmd_reg = UNSET_MUTRIG_DPATH_BIT_DUMMY_DATA_ENABLE(cmd_reg);
    if(dummy) {
        // set number of events per frame MUTRIG_DPATH_RANGE_DUMMY_DATA_CNT
        int n = n_;
        if(n_ > 255) n = 255;
        cmd_reg = SET_MUTRIG_DPATH_DUMMY_DATA_CNT_RANGE(cmd_reg, n);
        // set fast mode for dummy data generation MUTRIG_DPATH_BIT_DUMMY_DATA_FAST
        if(fast) cmd_reg =   SET_MUTRIG_DPATH_BIT_DUMMY_DATA_FAST(cmd_reg);
        else     cmd_reg = UNSET_MUTRIG_DPATH_BIT_DUMMY_DATA_FAST(cmd_reg);
    }
    write_register(MUTRIG_DPATH_REGISTER_W, cmd_reg); // reset config 
}

void MutrigDevice::start() {
  holdReset();                 // hold chip reset
  claimMem(DATA);              // claim mem interface
  //sync();                      // sync channels (not needed)
  chipReset();
  enable_continous_readout();  // setup dma, enable continous readout
  std::this_thread::sleep_for(std::chrono::microseconds(30)); // to make sure reset is on for 1 milli second
  releaseReset();              // release chip reset to start
}

void MutrigDevice::reset() {
  disable();
  releaseMem();
  releaseReset();
}

void MutrigDevice::stop() {
  holdReset();
  reset();
}
// channel reset -> sync
void MutrigDevice::sync() {
    uint32_t rst_reg;
    rst_reg = read_register_rw(RESET_REGISTER_W);
    // set reset in register for 100ns, only one cycle reset is sent to asics 
    write_register_wait(RESET_REGISTER_W,   SET_RESET_BIT_MUTRIG_CHANNEL(rst_reg), 100); 
    write_register(     RESET_REGISTER_W, UNSET_RESET_BIT_MUTRIG_CHANNEL(rst_reg));
} 

/*
 *
 */
void MutrigDevice::chipReset() {
    uint32_t rst_reg;
    rst_reg = read_register_rw(RESET_REGISTER_W);
    // set reset in register for 100ns, only one cycle reset is sent to asics 
    write_register_wait(RESET_REGISTER_W,   SET_RESET_BIT_MUTRIG_CHIP(rst_reg), 100); 
    write_register(     RESET_REGISTER_W, UNSET_RESET_BIT_MUTRIG_CHIP(rst_reg));
} 

// set chip reset
void MutrigDevice::holdReset() {
    uint32_t rst_reg;
    rst_reg = read_register_rw(RESET_REGISTER_W);
    rst_reg = SET_RESET_BIT_MUTRIG_CHANNEL_HOLD(rst_reg);
    write_register(     RESET_REGISTER_W, rst_reg);
    //std::cout << "DEBUG RESET " << rst_reg << " " << read_register_rw(RESET_REGISTER_W) << std::endl;
    //	print_registers();
}

// unset chip reset
void MutrigDevice::releaseReset() {
    uint32_t rst_reg;
    rst_reg = read_register_rw(RESET_REGISTER_W);
    rst_reg = UNSET_RESET_BIT_MUTRIG_CHANNEL_HOLD(rst_reg);
    write_register(RESET_REGISTER_W, rst_reg);
}

//Using a mutrig configuration instance holding the bitpattern, drive it to the ASIC:
//Write pattern to PCIe memory, trigger SPI configuration, Read back pattern to configuration instance.
//Throws on any hardware related errors (read/write/timeout), bitpattern checking is not done here.
void MutrigDevice::configureMUTRIG(mudaq::mutrig::Config&, int asic) {
    int err = 0;
    if( (err = isMemNot(SLOW)) )
        throw std::runtime_error("Mem interface is not in slow control state. Error code "+std::to_string(err));

    // check if "config done" is reset
    uint32_t cmd_reg;
    cmd_reg = read_register_ro(MUTRIG_CMD_REGISTER_R);
    if((GET_MUTRIG_CMD_BIT_CONFIG(cmd_reg)))
        throw std::runtime_error("'config done' is not reset. Reset asic slow control.");

    // write config bitpattern to mem
    for(uint i = 0; i < config.length_32bits; i++) {
        write_memory_rw(i, ((uint32_t*)config.bitpattern_w)[i]);
        uint32_t x = read_memory_rw(i);
        if(x != ((uint32_t*)config.bitpattern_w)[i])
            throw std::runtime_error("Couldn't write bit pattern to device");
    }

    //Set ASIC number and configuration bit to trigger SPI transaction
    cmd_reg = read_register_rw(MUTRIG_CMD_REGISTER_W);
    cmd_reg = SET_MUTRIG_CMD_HANDLE_RANGE(cmd_reg, asic);
    cmd_reg = SET_MUTRIG_CMD_BIT_CONFIG(cmd_reg);
    write_register(MUTRIG_CMD_REGISTER_W, cmd_reg);

    // check if config was written to the asic(s)
    // get config done
    uint cnt = 0;
    cmd_reg = read_register_ro(MUTRIG_CMD_REGISTER_R);
    while( (GET_MUTRIG_CMD_BIT_CONFIG(cmd_reg)) == 0) {
        if(++cnt >= 10000) throw std::runtime_error("Config was not written to the asic"+std::to_string(asic)+" within "+std::to_string(cnt)+"x 1ms");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        cmd_reg = read_register_ro(MUTRIG_CMD_REGISTER_R);
    }

    do{
        if(++cnt >= 10000) throw std::runtime_error("Config was not written to ASIC "+std::to_string(asic)+" within "+std::to_string(cnt)+"x 1ms");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
  	cmd_reg = read_register_ro(MUTRIG_CMD_REGISTER_R);
    }while( (GET_MUTRIG_CMD_BIT_CONFIG(cmd_reg)) == 0);
 
    //Reset configuration bit
    // this also resets config done, which re-arms cmd: config
    cmd_reg = read_register_rw(MUTRIG_CMD_REGISTER_W);
    cmd_reg = UNSET_MUTRIG_CMD_BIT_CONFIG(cmd_reg);
    write_register(MUTRIG_CMD_REGISTER_W, cmd_reg); // reset config


    // read back the settings from memory
    //Note: checking of readback value possible after, using config::VerifyReadbackPattern()
    for(uint i = 0; i < config.length_32bits; i++) {
        ((uint32_t*)config.bitpattern_r)[i] = read_memory_ro(i);
    }
    return;
}

} // namespace mudaq
