-- Register Map for slow contron memory on FEB

library ieee;
use ieee.std_logic_1164.all;

package feb_sc_registers is

    -- The block 0x00 is common for all FEBs
    -- and contains control and  monitoring information
    -- as well as the control of the slow control entity itself

    constant STATUS_REGISTER_R                  :   integer := 16#00#;
    constant GIT_HASH_REGISTER_R                :   integer := 16#01#;
    constant FPGA_TYPE_REGISTER_R               :   integer := 16#02#;
    constant FPGA_ID_REGISTER_RW                :   integer := 16#03#;
    constant CMD_LEN_REGISTER_RW                :   integer := 16#04#;
    constant CMD_OFFSET_REGISTER_RW             :   integer := 16#05#;
    constant RUN_STATE_RESET_BYPASS_REGISTER_RW :   integer := 16#06#;

    subtype RUN_STATE_RANGE                 is integer range 31 downto 16;
    subtype RESET_BYPASS_RANGE              is integer range 15 downto 0;

    constant RESET_PAYLOAD_RGEISTER_RW          :   integer := 16#07#;
    constant RESET_OPTICAL_LINKS_REGISTER_RW    :   integer := 16#08#;
    constant RESET_PHASE_REGISTER_R             :   integer := 16#09#;
    constant MERGER_RATE_REGISTER_R             :   integer := 16#0A#;

    constant ARRIA_TEMP_REGISTER_RW             :   integer := 16#10#;
    constant MAX10_ADC_0_1_REGISTER_R           :   integer := 16#11#;
    constant MAX10_ADC_2_3_REGISTER_R           :   integer := 16#12#;
    constant MAX10_ADC_4_5_REGISTER_R           :   integer := 16#13#;
    constant MAX10_ADC_6_7_REGISTER_R           :   integer := 16#14#;
    constant MAX10_ADC_8_9_REGISTER_R           :   integer := 16#15#;
    
    constant FIREFLY1_TEMP_REGISTER_R           :   integer := 16#16#;
    constant FIREFLY1_VOLT_REGISTER_R           :   integer := 16#17#;
    constant FIREFLY1_RX1_POW_REGISTER_R        :   integer := 16#18#;
    constant FIREFLY1_RX2_POW_REGISTER_R        :   integer := 16#19#;
    constant FIREFLY1_RX3_POW_REGISTER_R        :   integer := 16#1A#;
    constant FIREFLY1_RX4_POW_REGISTER_R        :   integer := 16#1B#;
    constant FIREFLY1_ALARM_REGISTER_R          :   integer := 16#1C#;

    constant FIREFLY2_TEMP_REGISTER_R           :   integer := 16#1D#;
    constant FIREFLY2_VOLT_REGISTER_R           :   integer := 16#1E#;
    constant FIREFLY2_RX1_POW_REGISTER_R        :   integer := 16#1F#;
    constant FIREFLY2_RX2_POW_REGISTER_R        :   integer := 16#20#;
    constant FIREFLY2_RX3_POW_REGISTER_R        :   integer := 16#21#;
    constant FIREFLY2_RX4_POW_REGISTER_R        :   integer := 16#22#;

    -- Registers 0x23 to 0x3F are reserved for further generic use
    -- Registers above 0x40 are for subdetector specific use

    subtype REG_AREA_RANGE is integer range 7 downto 6;
    constant REG_AREA_GENERIC : std_logic_vector(1 downto 0) := "00";

    -- MUPIX related registers
    constant BOARD_TH_W                         :   integer := 16#83#;
    subtype BOARD_TH_LOW_RANGE              is integer range 15 downto 0;
    subtype BOARD_TH_HIGH_RANGE             is integer range 31 downto 16;

    constant BOARD_INJECTION_W                  :   integer := 16#84#;
    subtype BOARD_INJECTION_RANGE           is integer range 15 downto 0;
    subtype BOARD_TH_PIX_RANGE              is integer range 31 downto 16;

    constant BOARD_TEMP_W                       :   integer := 16#85#;
    subtype BOARD_TEMP_DAC_RANGE            is integer range 15 downto 0;
    subtype BOARD_TEMP_ADC_RANGE            is integer range 31 downto 16;

    constant INJECTION1_OUT_A_FRONT_R           :   integer := 16#86#;
    constant THRESHOLD_PIX_OUT_A_FRONT_R        :   integer := 16#87#;
    constant THRESHOLD_LOW_OUT_A_FRONT_R        :   integer := 16#88#;   
    constant THRESHOLD_HIGH_OUT_A_FRONT_R       :   integer := 16#89#;
    constant BOARD_TEMP_DAC_OUT_R               :   integer := 16#8A#;
    constant BOARD_TEMP_ADC_OUT_R               :   integer := 16#8B#;
    constant A_SPI_WREN_FRONT_W                 :   integer := 16#8C#;
    constant CHIP_DAC_DATA_WE_W                 :   integer := 16#8D#;    
    constant CHIP_DAC_READY_W                   :   integer := 16#8E#;
    constant RESET_N_LVDS_W                     :   integer := 16#8F#;
    constant RO_PRESCALER_W                     :   integer := 16#90#;
    constant DEBUG_CHIP_SELECT_W                :   integer := 16#91#;
    constant TIMESTAMP_GRAY_INVERT_W            :   integer := 16#92#;
    constant MUX_READ_REGS_NIOS_W               :   integer := 16#93#;
    constant READ_REGS_MUPIX_MUX_R              :   integer := 16#94#;
    constant RESET_CHIP_DAC_FIFO_W              :   integer := 16#95#;
    constant LINK_MASK_REGISTER_RW              :   integer := 16#96#;
    constant LVDS_DATA_VALID_R                  :   integer := 16#97#; 
    constant LVDS_DATA_VALID_HI_R               :   integer := 16#9B#; 
    constant DISABLE_CONDITIONS_FOR_RUN_ACK_RW  :   integer := 16#98#;  
    constant CHIP_DACS_USEDW_R                  :   integer := 16#99#;
    constant REG_HITS_ENA_COUNT_R               :   integer := 16#9A#;
    constant SORTER_DELAY_RW                    :   integer := 16#A0#;


end package;
