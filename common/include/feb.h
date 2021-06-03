#ifndef __MU3E_HARDWARE_SC_H__
#define __MU3E_HARDWARE_SC_H__

namespace mu3e { namespace daq { namespace feb {

/* Mupix specific commands */
const uint16_t CMD_MUPIX_CHIP_CFG            = 0x0110;
const uint16_t CMD_MUPIX_BOARD_CFG           = 0x0120;


/*Timing detector common commands (mutrig)*/
const uint16_t CMD_MUTRIG_ASIC_CFG           = 0x0110; // configure ASIC # with pattern in payload. ASIC number is cmd&0x000F
/*commands 0x0110 ... 0x011f reserved*/
const uint16_t CMD_MUTRIG_ASIC_OFF           = 0x0103; // configure all off builtin pattern
const uint16_t CMD_MUTRIG_CNT_READ           = 0x0105;
const uint16_t CMD_MUTRIG_CNT_RESET          = 0x0106;
const uint16_t CMD_MUTRIG_SKEW_RESET         = 0x0104;

/* Tiles specific commands */
const uint16_t CMD_TILE_ON                   = 0x0101; // automatic powering routine of TMB (maskbit-defined)
const uint16_t CMD_TILE_OFF                  = 0x0102; // automatic powering routine of TMB (maskbit-defined)
const uint16_t CMD_TILE_TEMPERATURES_READ    = 0x0103;
const uint16_t CMD_TILE_POWERMONITORS_READ   = 0x0104;
const uint16_t CMD_TILE_TMB_ON               = 0x0105;
const uint16_t CMD_TILE_TMB_OFF              = 0x0106;
const uint16_t CMD_TILE_ASIC_ON              = 0x0120; // switch on  ASIC #. ASIC number is cmd&0x000F
const uint16_t CMD_TILE_ASIC_OFF             = 0x0130; // switch off ASIC #. ASIC number is cmd&0x000F

//const uint16_t CMD_TILE_I2C_WRITE            = 0x0105;

/* SciFi specific commands */
//Fill me

/* FEB common commands */
const uint16_t CMD_PING                      = 0xFFFE;
const uint16_t CMD_FFFF                      = 0xFFFF;

inline
uint32_t make_cmd(uint16_t cmd, uint16_t n = 0) {
    return ((uint32_t)cmd << 16) | ((uint32_t)n & 0xFFFF);
}

} } } // namespace mu3e::daq::feb

#endif // __MU3E_HARDWARE_SC_H__
