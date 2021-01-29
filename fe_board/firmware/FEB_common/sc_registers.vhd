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
    constant FIREFLY2_ALARM_REGISTER_R          :   integer := 16#23#;

    constant NONINCREMENTING_TEST_REGISTER_RW   :   integer := 16#24#;

    -- Registers 0x25 to 0x3F are reserved for further generic use
    -- Registers above 0x40 are for subdetector specific use

    subtype REG_AREA_RANGE is integer range 7 downto 6;
    constant REG_AREA_GENERIC : std_logic_vector(1 downto 0) := "00";

end package;
