-- Register Map for slow contron memory on FEB

library ieee;
use ieee.std_logic_1164.all;

package feb_sc_registers is

    constant PACKET_TYPE_SC                        : std_logic_vector(2 downto 0) := "111";
    
    constant PACKET_TYPE_SC_READ                   : std_logic_vector(1 downto 0) := "00";
    constant PACKET_TYPE_SC_WRITE                  : std_logic_vector(1 downto 0) := "01";
    constant PACKET_TYPE_SC_READ_NONINCREMENTING   : std_logic_vector(1 downto 0) := "10";
    constant PACKET_TYPE_SC_WRITE_NONINCREMENTING  : std_logic_vector(1 downto 0) := "11";

    subtype FEB_SC_ADDR_RANGE       is integer range 255 downto 0;
    subtype FEB_SC_DATA_SIZE_RANGE  is integer range 512 downto 1;
    constant FEB_SC_RAM_SIZE : std_logic_vector(3 downto 0) := "1110";

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

    subtype  RUN_STATE_RANGE                    is  integer range 31 downto 16;
    subtype  RESET_BYPASS_RANGE                 is  integer range 7 downto 0;
    constant RESET_BYPASS_BIT_REQUEST           :   integer := 8;
    constant RESET_BYPASS_BIT_ENABLE            :   integer := 9;

    constant RESET_PAYLOAD_REGISTER_RW          :   integer := 16#07#;
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
    constant FIREFLY2_ALARM_REGISTER_R          :   integer := 16#23#;

    constant NONINCREMENTING_TEST_REGISTER_RW   :   integer := 16#24#;
    constant MAX10_VERSION_REGISTER_R           :   integer := 16#25#;
    constant MAX10_STATUS_REGISTER_R            :   integer := 16#26#;
    constant MAX10_STATUS_BIT_PLL_LOCKED        :   integer := 0;
    constant MAX10_STATUS_BIT_SPI_ARRIA_CLK     :   integer := 1;
    constant PROGRAMMING_CTRL_W                 :   integer := 16#27#;
    constant PROGRAMmING_STATUS_R               :   integer := 16#28#;
    constant PROGRAMMING_ADDR_W                 :   integer := 16#29#;  
    constant PROGRAMMING_DATA_W                 :   integer := 16#2A#;


    -- Registers 0x25 to 0x3F are reserved for further generic use
    -- Registers above 0x40 are for subdetector specific use

    subtype REG_AREA_RANGE is integer range 7 downto 6;
    constant REG_AREA_GENERIC : std_logic_vector(1 downto 0) := "00";

    -- MUPIX related registers
    constant LVDS_PLL_LOCKED_REGISTER_R         : integer := 16#40#;
    constant LINK_MASK_REGISTER_W               : integer := 16#41#;
    constant TIMESTAMP_GRAY_INVERT_REGISTER_W   : integer := 16#42#;
    constant TS_INVERT_BIT                      : integer := 0;
    constant TS2_INVERT_BIT                     : integer := 1;
    constant TS_GRAY_BIT                        : integer := 2;
    constant TS2_GRAY_BIT                       : integer := 3;
    constant DEBUG_CHIP_SELECT_REGISTER_W       : integer := 16#43#;
    constant RO_PRESCALER_REGISTER_W            : integer := 16#44#;
    constant MULTICHIP_RO_OVERFLOW_REGISTER_R   : integer := 16#4F#;
    constant LVDS_RUNCOUNTER_REGISTER_R         : integer := 16#50#; --Dec 80-115
    constant LVDS_ERRCOUNTER_REGISTER_R         : integer := 16#74#; --Dec 116-151

    constant BOARD_TH_W                         :   integer := 16#A3#;
    subtype BOARD_TH_LOW_RANGE              is integer range 15 downto 0;
    subtype BOARD_TH_HIGH_RANGE             is integer range 31 downto 16;

    constant BOARD_INJECTION_W                  :   integer := 16#A4#;
    subtype BOARD_INJECTION_RANGE           is integer range 15 downto 0;
    subtype BOARD_TH_PIX_RANGE              is integer range 31 downto 16;

    constant BOARD_TEMP_W                       :   integer := 16#A5#;
    subtype BOARD_TEMP_DAC_RANGE            is integer range 15 downto 0;
    subtype BOARD_TEMP_ADC_RANGE            is integer range 31 downto 16;

    constant NREGISTERS_MUPIX_WR                : integer := 256;
    constant NREGISTERS_MUPIX_RD                : integer := 256;

    constant INJECTION1_OUT_A_FRONT_R           :   integer := 16#A6#;
    constant THRESHOLD_PIX_OUT_A_FRONT_R        :   integer := 16#A7#;
    constant THRESHOLD_LOW_OUT_A_FRONT_R        :   integer := 16#A8#;   
    constant THRESHOLD_HIGH_OUT_A_FRONT_R       :   integer := 16#A9#;
    constant BOARD_TEMP_DAC_OUT_R               :   integer := 16#AA#;
    constant BOARD_TEMP_ADC_OUT_R               :   integer := 16#AB#;
    constant A_SPI_WREN_FRONT_W                 :   integer := 16#AC#;
    constant CHIP_DAC_DATA_WE_W                 :   integer := 16#AD#;    
    constant CHIP_DAC_READY_W                   :   integer := 16#AE#;
    constant RESET_N_LVDS_W                     :   integer := 16#AF#;

    constant MUX_READ_REGS_NIOS_W               :   integer := 16#B3#;
    constant READ_REGS_MUPIX_MUX_R              :   integer := 16#B4#;
    constant RESET_CHIP_DAC_FIFO_W              :   integer := 16#B5#;
    constant LINK_MASK_REGISTER_RW              :   integer := 16#B6#;
    constant LVDS_DATA_VALID_R                  :   integer := 16#B7#; 
    constant LVDS_DATA_VALID_HI_R               :   integer := 16#BB#; 
    constant DISABLE_CONDITIONS_FOR_RUN_ACK_RW  :   integer := 16#B8#;  
    constant CHIP_DACS_USEDW_R                  :   integer := 16#B9#;
    constant REG_HITS_ENA_COUNT_R               :   integer := 16#BA#;
    constant SORTER_DELAY_RW                    :   integer := 16#BF#;
    constant SORTER_COUNTER_R                   :   integer := 16#C0#; 
    -- 40 counters
    -- next free register at 16#E8#


end package;
