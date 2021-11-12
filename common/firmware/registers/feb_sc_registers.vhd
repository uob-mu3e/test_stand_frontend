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

    constant STATUS_REGISTER_R                  :   integer := 16#FC00#;        -- DOC: not in use | FEB_ALL
    constant GIT_HASH_REGISTER_R                :   integer := 16#FC01#;        -- DOC: contains the git hash of used firmware | FEB_ALL
    constant FPGA_TYPE_REGISTER_R               :   integer := 16#FC02#;        -- DOC: contains fpga type 111010: mupix, 111000 : mutrig | FEB_ALL
    constant FPGA_ID_REGISTER_RW                :   integer := 16#FC03#;        -- DOC: fgpa ID | FEB_ALL
    constant CMD_LEN_REGISTER_RW                :   integer := 16#FC04#;        -- DOC: length of data to read in a nios rpc call | FEB_ALL
    constant CMD_OFFSET_REGISTER_RW             :   integer := 16#FC05#;        -- DOC: positon of the data to read in a nios rpc call | FEB_ALL
    constant RUN_STATE_RESET_BYPASS_REGISTER_RW :   integer := 16#FC06#;        -- DOC: used to bypass the reset system and run without a clock box | FEB_ALL

    subtype  RUN_STATE_RANGE                    is  integer range 31 downto 16; -- DOC: not in use | FEB_ALL
    subtype  RESET_BYPASS_RANGE                 is  integer range 7 downto 0;   -- DOC: not in use | FEB_ALL
    constant RESET_BYPASS_BIT_REQUEST           :   integer := 8;               -- DOC: sends a bypass reset command | FEB_ALL
    constant RESET_BYPASS_BIT_ENABLE            :   integer := 9;               -- DOC: enables the use of the reset bypass | FEB_ALL

    constant RESET_PAYLOAD_REGISTER_RW          :   integer := 16#FC07#;        -- DOC: payload for the reset bypass commands | FEB_ALL
    constant RESET_OPTICAL_LINKS_REGISTER_RW    :   integer := 16#FC08#;        -- DOC: reset firefly | FEB_ALL
    constant RESET_PHASE_REGISTER_R             :   integer := 16#FC09#;        -- DOC: phase between reset rx clock and global clk | FEB_ALL
    constant MERGER_RATE_REGISTER_R             :   integer := 16#FC0A#;        -- DOC: output rate of the data merger | FEB_ALL

    constant ARRIA_TEMP_REGISTER_RW             :   integer := 16#FC10#;        -- DOC: ARRIAV internal temp sense | FEB_ALL
    constant MAX10_ADC_0_1_REGISTER_R           :   integer := 16#FC11#;        -- DOC: MAX10 adc data | FEB_ALL
    constant MAX10_ADC_2_3_REGISTER_R           :   integer := 16#FC12#;
    constant MAX10_ADC_4_5_REGISTER_R           :   integer := 16#FC13#;
    constant MAX10_ADC_6_7_REGISTER_R           :   integer := 16#FC14#;
    constant MAX10_ADC_8_9_REGISTER_R           :   integer := 16#FC15#;
    
    constant FIREFLY1_TEMP_REGISTER_R           :   integer := 16#FC16#;        -- DOC: contains the temp of firefly1 | FEB_ALL
    constant FIREFLY1_VOLT_REGISTER_R           :   integer := 16#FC17#;        -- DOC: contains the voltage of firefly1 | FEB_ALL
    constant FIREFLY1_RX1_POW_REGISTER_R        :   integer := 16#FC18#;        -- DOC: received optical power on link 1 | FEB_ALL
    constant FIREFLY1_RX2_POW_REGISTER_R        :   integer := 16#FC19#;        -- DOC: received optical power on link 2 | FEB_ALL
    constant FIREFLY1_RX3_POW_REGISTER_R        :   integer := 16#FC1A#;        -- DOC: received optical power on link 3 | FEB_ALL
    constant FIREFLY1_RX4_POW_REGISTER_R        :   integer := 16#FC1B#;        -- DOC: received optical power on link 4 | FEB_ALL
    constant FIREFLY1_ALARM_REGISTER_R          :   integer := 16#FC1C#;        -- DOC: firefly1 alarm | FEB_ALL

    constant FIREFLY2_TEMP_REGISTER_R           :   integer := 16#FC1D#;        -- DOC: contains the temp of firefly2 | FEB_ALL
    constant FIREFLY2_VOLT_REGISTER_R           :   integer := 16#FC1E#;        -- DOC: contains the voltage of firefly2 | FEB_ALL
    constant FIREFLY2_RX1_POW_REGISTER_R        :   integer := 16#FC1F#;        -- DOC: received optical power on link 5 | FEB_ALL
    constant FIREFLY2_RX2_POW_REGISTER_R        :   integer := 16#FC20#;        -- DOC: received optical power on link 6 | FEB_ALL
    constant FIREFLY2_RX3_POW_REGISTER_R        :   integer := 16#FC21#;        -- DOC: received optical power on link 7 | FEB_ALL
    constant FIREFLY2_RX4_POW_REGISTER_R        :   integer := 16#FC22#;        -- DOC: received optical power on link 8 | FEB_ALL
    constant FIREFLY2_ALARM_REGISTER_R          :   integer := 16#FC23#;        -- DOC: firefly1 alarm | FEB_ALL

    constant NONINCREMENTING_TEST_REGISTER_RW   :   integer := 16#FC24#;        -- DOC: testing nonincrementing reads/writes to fifo (not in use) | FEB_ALL
    constant MAX10_VERSION_REGISTER_R           :   integer := 16#FC25#;        -- DOC: git hash of the max10 firmware | FEB_ALL
    constant MAX10_STATUS_REGISTER_R            :   integer := 16#FC26#;
    constant MAX10_STATUS_BIT_PLL_LOCKED        :   integer := 0;
    constant MAX10_STATUS_BIT_SPI_ARRIA_CLK     :   integer := 1;
    constant PROGRAMMING_CTRL_W                 :   integer := 16#FC27#;
    constant PROGRAMmING_STATUS_R               :   integer := 16#FC28#;
    constant PROGRAMMING_ADDR_W                 :   integer := 16#FC29#;  
    constant PROGRAMMING_DATA_W                 :   integer := 16#FC2A#;

    constant SI_STATUS_REGISTER                 :   integer := 16#FC2B#;

end package;
