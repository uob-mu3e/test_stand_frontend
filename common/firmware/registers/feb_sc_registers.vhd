-- Register Map for common parts of the sc memory on ArriaV
-- addr range: 0xFC00-0xFFFE

library ieee;
use ieee.std_logic_1164.all;

package feb_sc_registers is

    constant PACKET_TYPE_SC                        : std_logic_vector(2 downto 0) := "111";
    
    constant PACKET_TYPE_SC_READ                   : std_logic_vector(1 downto 0) := "00";
    constant PACKET_TYPE_SC_WRITE                  : std_logic_vector(1 downto 0) := "01";
    constant PACKET_TYPE_SC_READ_NONINCREMENTING   : std_logic_vector(1 downto 0) := "10";
    constant PACKET_TYPE_SC_WRITE_NONINCREMENTING  : std_logic_vector(1 downto 0) := "11";

    -- TODO: software checks
    --subtype FEB_SC_ADDR_RANGE       is integer range 255 downto 0;
    --subtype FEB_SC_DATA_SIZE_RANGE  is integer range 512 downto 1;
    --constant FEB_SC_RAM_SIZE : std_logic_vector(3 downto 0) := "1110";

    constant STATUS_REGISTER_R                  :   integer := 16#FC00#;
    constant GIT_HASH_REGISTER_R                :   integer := 16#FC01#;
    constant FPGA_TYPE_REGISTER_R               :   integer := 16#FC02#;
    constant FPGA_ID_REGISTER_RW                :   integer := 16#FC03#;
    constant CMD_LEN_REGISTER_RW                :   integer := 16#FC04#;
    constant CMD_OFFSET_REGISTER_RW             :   integer := 16#FC05#;
    constant RUN_STATE_RESET_BYPASS_REGISTER_RW :   integer := 16#FC06#;

    subtype  RUN_STATE_RANGE                    is  integer range 31 downto 16;
    subtype  RESET_BYPASS_RANGE                 is  integer range 7 downto 0;
    constant RESET_BYPASS_BIT_REQUEST           :   integer := 8;
    constant RESET_BYPASS_BIT_ENABLE            :   integer := 9;

    constant RESET_PAYLOAD_REGISTER_RW          :   integer := 16#FC07#;
    constant RESET_OPTICAL_LINKS_REGISTER_RW    :   integer := 16#FC08#;
    constant RESET_PHASE_REGISTER_R             :   integer := 16#FC09#;
    constant MERGER_RATE_REGISTER_R             :   integer := 16#FC0A#;

    constant ARRIA_TEMP_REGISTER_RW             :   integer := 16#FC10#;
    constant MAX10_ADC_0_1_REGISTER_R           :   integer := 16#FC11#;
    constant MAX10_ADC_2_3_REGISTER_R           :   integer := 16#FC12#;
    constant MAX10_ADC_4_5_REGISTER_R           :   integer := 16#FC13#;
    constant MAX10_ADC_6_7_REGISTER_R           :   integer := 16#FC14#;
    constant MAX10_ADC_8_9_REGISTER_R           :   integer := 16#FC15#;
    
    constant FIREFLY1_TEMP_REGISTER_R           :   integer := 16#FC16#;
    constant FIREFLY1_VOLT_REGISTER_R           :   integer := 16#FC17#;
    constant FIREFLY1_RX1_POW_REGISTER_R        :   integer := 16#FC18#;
    constant FIREFLY1_RX2_POW_REGISTER_R        :   integer := 16#FC19#;
    constant FIREFLY1_RX3_POW_REGISTER_R        :   integer := 16#FC1A#;
    constant FIREFLY1_RX4_POW_REGISTER_R        :   integer := 16#FC1B#;
    constant FIREFLY1_ALARM_REGISTER_R          :   integer := 16#FC1C#;

    constant FIREFLY2_TEMP_REGISTER_R           :   integer := 16#FC1D#;
    constant FIREFLY2_VOLT_REGISTER_R           :   integer := 16#FC1E#;
    constant FIREFLY2_RX1_POW_REGISTER_R        :   integer := 16#FC1F#;
    constant FIREFLY2_RX2_POW_REGISTER_R        :   integer := 16#FC20#;
    constant FIREFLY2_RX3_POW_REGISTER_R        :   integer := 16#FC21#;
    constant FIREFLY2_RX4_POW_REGISTER_R        :   integer := 16#FC22#;
    constant FIREFLY2_ALARM_REGISTER_R          :   integer := 16#FC23#;

    constant NONINCREMENTING_TEST_REGISTER_RW   :   integer := 16#FC24#;
    constant MAX10_VERSION_REGISTER_R           :   integer := 16#FC25#;
    constant MAX10_STATUS_REGISTER_R            :   integer := 16#FC26#;
    constant MAX10_STATUS_BIT_PLL_LOCKED        :   integer := 0;
    constant MAX10_STATUS_BIT_SPI_ARRIA_CLK     :   integer := 1;
    constant PROGRAMMING_CTRL_W                 :   integer := 16#FC27#;
    constant PROGRAMmING_STATUS_R               :   integer := 16#FC28#;
    constant PROGRAMMING_ADDR_W                 :   integer := 16#FC29#;  
    constant PROGRAMMING_DATA_W                 :   integer := 16#FC2A#;

    constant SI_STATUS_REGISTER                 :   integer := 16#FC2B#;

end package;
