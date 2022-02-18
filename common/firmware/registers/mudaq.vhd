
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use std.textio.all;
use ieee.std_logic_textio.all;


package mudaq is

--! 8b/10b words
    constant D16_2 : std_logic_vector(7 downto 0) := X"50";
    constant D21_4 : std_logic_vector(7 downto 0) := x"95";
    constant D02_5 : std_logic_vector(7 downto 0) := X"A2";
    constant D21_5 : std_logic_vector(7 downto 0) := x"B5";
    constant D28_4 : std_logic_vector(7 downto 0) := x"9C";
    constant D28_5 : std_logic_vector(7 downto 0) := X"BC";
    constant D28_7 : std_logic_vector(7 downto 0) := X"FC";
    constant D05_6 : std_logic_vector(7 downto 0) := X"C5";
    constant K28_0 : std_logic_vector(7 downto 0) := X"1C"; -- still used in MuPix ??
    constant K28_1 : std_logic_vector(7 downto 0) := X"3C"; -- still used in data alignment (transceiver) ??
    constant K28_2 : std_logic_vector(7 downto 0) := X"5C";
    constant K28_3 : std_logic_vector(7 downto 0) := X"7C";
    constant K28_4 : std_logic_vector(7 downto 0) := X"9C"; -- used as end of packet marker between FEB <--> SW board
    constant K28_5 : std_logic_vector(7 downto 0) := X"BC"; -- still used in MuPix ???
    constant K28_6 : std_logic_vector(7 downto 0) := X"DC";
    constant K28_7 : std_logic_vector(7 downto 0) := X"FC"; -- not used, comma symbol with harder constraints!
    constant K23_7 : std_logic_vector(7 downto 0) := X"F7"; -- still used as "empty" data (transceiver) ??
    constant K27_7 : std_logic_vector(7 downto 0) := X"FB";
    constant K29_7 : std_logic_vector(7 downto 0) := X"FD";
    constant K30_7 : std_logic_vector(7 downto 0) := X"FE";


    --! data path farm types
    subtype dataplusts_type is std_logic_vector(519 downto 0);
    type offset is array(natural range <>) of integer range 64 downto 0;
    subtype tsrange_type is std_logic_vector(7 downto 0);
    subtype tsupper is natural range 15 downto 8;
    subtype tslower is natural range 7 downto 0;
    constant tsone : tsrange_type := (others => '1');
    constant tszero : tsrange_type := (others => '0');


    --! time merger tree constants
    constant IDEL : std_logic_vector(3 downto 0)    := x"0";
    constant ONEMASK : std_logic_vector(3 downto 0) := x"1";
    constant WAITING : std_logic_vector(3 downto 0) := x"2";
    constant HEADER : std_logic_vector(3 downto 0)  := x"3";
    constant TS0 : std_logic_vector(3 downto 0)     := x"4";
    constant TS1 : std_logic_vector(3 downto 0)     := x"5";
    constant SHEADER : std_logic_vector(3 downto 0) := x"6";
    constant HIT : std_logic_vector(3 downto 0)     := x"7";
    constant ONEHIT : std_logic_vector(3 downto 0)  := x"8";
    constant TRAILER : std_logic_vector(3 downto 0) := x"9";
    constant ONEERROR : std_logic_vector(3 downto 0) := x"a";
    type N_LINKS_TREE_t is array (3 downto 0) of integer;
    constant N_LINKS_TREE : N_LINKS_TREE_t := (1, 2, 4, 8);


    --! FEB - SWB protocol
    constant HEADER_K:    std_logic_vector(31 downto 0) := x"bcbcbcbc";
    constant DATA_HEADER_ID:    std_logic_vector(5 downto 0) := "111010";
    constant DATA_SUB_HEADER_ID:    std_logic_vector(5 downto 0) := "111111";
    constant ACTIVE_SIGNAL_HEADER_ID:    std_logic_vector(5 downto 0) := "111101";
    constant RUN_TAIL_HEADER_ID:    std_logic_vector(5 downto 0) := "111110";
    constant TIMING_MEAS_HEADER_ID:    std_logic_vector(5 downto 0) := "111100";
    constant SC_HEADER_ID:    std_logic_vector(5 downto 0) := "111011";
    constant PREAMBLE_TYPE_MUPIX_c  : std_logic_vector(5 downto 0) := "111010";
    constant PREAMBLE_TYPE_MUTRIG_c : std_logic_vector(5 downto 0) := "111000";
    constant PREAMBLE_TYPE_SC_c     : std_logic_vector(5 downto 0) := "000111";
    constant PREAMBLE_TYPE_BERT_c   : std_logic_vector(5 downto 0) := "000010";
    constant PREAMBLE_TYPE_IDLE_c   : std_logic_vector(5 downto 0) := "000000";

    -- out of band
    constant SC_OOB_c       : std_logic_vector(1 downto 0) := "00";
    constant SC_READ_c      : std_logic_vector(1 downto 0) := "10";
    constant SC_WRITE_c     : std_logic_vector(1 downto 0) := "11";

    -- start of packet
    constant FIFO_SOP_c     : std_logic_vector(3 downto 0) := "0010";
    -- payload
    constant FIFO_PAYLOAD_c : std_logic_vector(3 downto 0) := "0000";
    -- end of packet
    constant FIFO_EOP_c     : std_logic_vector(3 downto 0) := "0011";
    -- end of run
    constant FIFO_EOR_c     : std_logic_vector(3 downto 0) := "0111";


    subtype reg32 is std_logic_vector(31 downto 0);
    subtype reg24 is std_logic_vector(23 downto 0);
    subtype reg16 is std_logic_vector(15 downto 0);
    subtype reg8  is std_logic_vector( 7 downto 0);
    subtype reg4  is std_logic_vector( 3 downto 0);

    -- TODO: REMOVE THIS NUMBER HERE
    constant NREGISTERS :  integer := 64;
    type reg32array_pcie is array (NREGISTERS-1 downto 0) of reg32;

    type reg32array is array (natural range <>) of reg32;
    type reg24array is array (natural range <>) of reg24;
    type reg16array is array (natural range <>) of reg16;
    type  reg8array is array (natural range <>) of  reg8;
    type  reg4array is array (natural range <>) of  reg4;

    --! type for run state
    subtype run_state_t is std_logic_vector(9 downto 0);

    constant RUN_STATE_BITPOS_IDLE        : natural := 0;
    constant RUN_STATE_BITPOS_PREP        : natural := 1;
    constant RUN_STATE_BITPOS_SYNC        : natural := 2;
    constant RUN_STATE_BITPOS_RUNNING     : natural := 3;
    constant RUN_STATE_BITPOS_TERMINATING : natural := 4;
    constant RUN_STATE_BITPOS_LINK_TEST   : natural := 5;
    constant RUN_STATE_BITPOS_SYNC_TEST   : natural := 6;
    constant RUN_STATE_BITPOS_RESET       : natural := 7;
    constant RUN_STATE_BITPOS_OUT_OF_DAQ  : natural := 8;

    constant RUN_STATE_IDLE        : run_state_t := (RUN_STATE_BITPOS_IDLE         => '1', others =>'0');
    constant RUN_STATE_PREP        : run_state_t := (RUN_STATE_BITPOS_PREP         => '1', others =>'0');
    constant RUN_STATE_SYNC        : run_state_t := (RUN_STATE_BITPOS_SYNC         => '1', others =>'0');
    constant RUN_STATE_RUNNING     : run_state_t := (RUN_STATE_BITPOS_RUNNING      => '1', others =>'0');
    constant RUN_STATE_TERMINATING : run_state_t := (RUN_STATE_BITPOS_TERMINATING  => '1', others =>'0');
    constant RUN_STATE_LINK_TEST   : run_state_t := (RUN_STATE_BITPOS_LINK_TEST    => '1', others =>'0');
    constant RUN_STATE_SYNC_TEST   : run_state_t := (RUN_STATE_BITPOS_SYNC_TEST    => '1', others =>'0');
    constant RUN_STATE_RESET       : run_state_t := (RUN_STATE_BITPOS_RESET        => '1', others =>'0');
    constant RUN_STATE_OUT_OF_DAQ  : run_state_t := (RUN_STATE_BITPOS_OUT_OF_DAQ   => '1', others =>'0');
    constant RUN_STATE_UNDEFINED   : run_state_t := (others =>'0');

    type feb_run_state is (
        idle,
        run_prep,
        sync,
        running,
        terminating,
        link_test,
        sync_test,
        reset_state,
        out_of_DAQ
    );

    -- reset link words
    constant RESET_LINK_RUN_PREPARE : std_logic_vector(7 DOWNTO 0) := x"10";
    constant RESET_LINK_SYNC : std_logic_vector(7 DOWNTO 0) := x"11";
    constant RESET_LINK_START_RUN : std_logic_vector(7 DOWNTO 0) := x"12";
    constant RESET_LINK_END_RUN : std_logic_vector(7 DOWNTO 0) := x"13";
    constant RESET_LINK_ABORT_RUN : std_logic_vector(7 DOWNTO 0) := x"14";
    constant RESET_LINK_START_LINK_TEST : std_logic_vector(7 DOWNTO 0) := x"20";
    constant RESET_LINK_STOP_LINK_TEST : std_logic_vector(7 DOWNTO 0) := x"21";
    constant RESET_LINK_START_SYNC_TEST : std_logic_vector(7 DOWNTO 0) := x"24";
    constant RESET_LINK_STOP_SYNC_TEST : std_logic_vector(7 DOWNTO 0) := x"25";
    constant RESET_LINK_TEST_SYNC : std_logic_vector(7 DOWNTO 0) := x"26";
    constant RESET_LINK_RESET : std_logic_vector(7 DOWNTO 0) := x"30";
    constant RESET_LINK_STOP_RESET : std_logic_vector(7 DOWNTO 0) := x"31";
    constant RESET_LINK_ENABLE : std_logic_vector(7 DOWNTO 0) := x"32";
    constant RESET_LINK_DISABLE : std_logic_vector(7 DOWNTO 0) := x"33";
    constant RESET_LINK_ADDRESS : std_logic_vector(7 DOWNTO 0) := x"34";


    -- time constants
    constant TIME_125MHz_1s     : std_logic_vector(27 DOWNTO 0) := x"7735940";
    constant TIME_125MHz_1ms    : std_logic_vector(27 DOWNTO 0) := x"001E848";
    constant TIME_125MHz_2s     : std_logic_vector(27 DOWNTO 0) := x"EE6B280";
    constant HUNDRED_MILLION    : std_logic_vector(27 downto 0) := x"5F5E100";
    constant HUNDRED_MILLION32  : std_logic_vector(31 downto 0) := x"05F5E100";


    -- mscb addressing (for networks with 8bit and 16bit addresses, we will use 16 ?)
    constant MSCB_CMD_ADDR_NODE16           : std_logic_vector(7 downto 0)      := X"0A";
    constant MSCB_CMD_ADDR_NODE8            : std_logic_vector(7 downto 0)      := X"09";
    constant MSCB_CMD_ADDR_GRP8             : std_logic_vector(7 downto 0)      := X"11"; -- group addressing
    constant MSCB_CMD_ADDR_GRP16            : std_logic_vector(7 downto 0)      := X"12";
    constant MSCB_CMD_ADDR_BC               : std_logic_vector(7 downto 0)      := X"10"; --broadcast
    constant MSCB_CMD_PING8                 : std_logic_vector(7 downto 0)      := X"19";
    constant MSCB_CMD_PING16                : std_logic_vector(7 downto 0)      := X"1A";

    constant run_prep_acknowledge           : std_logic_vector(31 downto 0)     := x"000000FE";
    constant run_prep_acknowledge_datak     : std_logic_vector(3 downto 0)      := "0001";
    constant RUN_END                        : std_logic_vector(31 downto 0)     := x"000000FD";
    constant RUN_END_DATAK                  : std_logic_vector(3 downto 0)      := "0001";
    constant MERGER_TIMEOUT                 : std_logic_vector(31 downto 0)     := x"000000FB";
    constant MERGER_TIMEOUT_DATAK           : std_logic_vector(3 downto 0)      := "0001";

    constant MERGER_FIFO_RUN_END_MARKER     : std_logic_vector(3 downto 0)      := "0111";
    constant MERGER_FIFO_PAKET_END_MARKER   : std_logic_vector(3 downto 0)      := "0011";
    constant MERGER_FIFO_PAKET_START_MARKER : std_logic_vector(3 downto 0)      := "0010";

    -- FEB Arria-MAX SPI addresses
    constant FEBSPI_ADDR_GITHASH            : std_logic_vector(6 downto 0)      := "0000000";
    constant FEBSPI_ADDR_STATUS             : std_logic_vector(6 downto 0)      := "0000010";
    constant FEBSPI_ADDR_CONTROL            : std_logic_vector(6 downto 0)      := "0000011";
    constant FEBSPI_ADDR_RESET              : std_logic_vector(6 downto 0)      := "0000100";
    constant FEBSPI_ADDR_PROGRAMMING_STATUS : std_logic_vector(6 downto 0)      := "0010000";
    constant FEBSPI_ADDR_PROGRAMMING_COUNT  : std_logic_vector(6 downto 0)      := "0010001";
    constant FEBSPI_ADDR_PROGRAMMING_CTRL   : std_logic_vector(6 downto 0)      := "0010010";
    constant FEBSPI_ADDR_PROGRAMMING_ADDR   : std_logic_vector(6 downto 0)      := "0010011";
    constant FEBSPI_ADDR_PROGRAMMING_WFIFO  : std_logic_vector(6 downto 0)      := "0010100";

    constant FEBSPI_ADDR_ADCCTRL            : std_logic_vector(6 downto 0)      := "0100000";
    constant FEBSPI_ADDR_ADCDATA            : std_logic_vector(6 downto 0)      := "0100001";


    -- FEB-MAX SPI Flash
    constant COMMAND_WRITE_ENABLE               : std_logic_vector(7 downto 0) := X"06";
    constant COMMAND_WRITE_DISABLE              : std_logic_vector(7 downto 0) := X"04";
    constant COMMAND_READ_STATUS_REGISTER1      : std_logic_vector(7 downto 0) := X"05";
    constant COMMAND_READ_STATUS_REGISTER2      : std_logic_vector(7 downto 0) := X"35";   
    constant COMMAND_READ_STATUS_REGISTER3      : std_logic_vector(7 downto 0) := X"15";
    constant COMMAND_WRITE_ENABLE_VSR           : std_logic_vector(7 downto 0) := X"50";
    constant COMMAND_WRITE_STATUS_REGISTER1     : std_logic_vector(7 downto 0) := X"01";
    constant COMMAND_WRITE_STATUS_REGISTER2     : std_logic_vector(7 downto 0) := X"31";   
    constant COMMAND_WRITE_STATUS_REGISTER3     : std_logic_vector(7 downto 0) := X"11";
    constant COMMAND_READ_DATA                  : std_logic_vector(7 downto 0) := X"03";
    constant COMMAND_FAST_READ                  : std_logic_vector(7 downto 0) := X"0B";
    constant COMMAND_DUAL_OUTPUT_FAST_READ      : std_logic_vector(7 downto 0) := X"3B";
    constant COMMAND_DUAL_IO_FAST_READ          : std_logic_vector(7 downto 0) := X"BB";
    constant COMMAND_QUAD_OUTPUT_FAST_READ      : std_logic_vector(7 downto 0) := X"6B";
    constant COMMAND_QUAD_IO_FAST_READ          : std_logic_vector(7 downto 0) := X"EB";
    constant COMMAND_QUAD_IO_WORD_FAST_READ     : std_logic_vector(7 downto 0) := X"E7";
    constant COMMAND_PAGE_PROGRAM               : std_logic_vector(7 downto 0) := X"02";
    constant COMMAND_QUAD_PAGE_PROGRAM          : std_logic_vector(7 downto 0) := X"32";
    constant COMMAND_FAST_PAGE_PROGRAM          : std_logic_vector(7 downto 0) := X"F2"; 
    constant COMMAND_SECTOR_ERASE               : std_logic_vector(7 downto 0) := X"20";
    constant COMMAND_BLOCK_ERASE_32             : std_logic_vector(7 downto 0) := X"52";
    constant COMMAND_BLOCK_ERASE_64             : std_logic_vector(7 downto 0) := X"D8";
    constant COMMAND_CHIP_ERASE                 : std_logic_vector(7 downto 0) := X"C7";
    constant COMMAND_ENABLE_RESET               : std_logic_vector(7 downto 0) := X"66";
    constant COMMAND_RESET                      : std_logic_vector(7 downto 0) := X"99";
    constant COMMAND_JEDEC_ID                   : std_logic_vector(7 downto 0) := X"9F"; 
    constant COMMAND_ERASE_SECURITY_REGISTERS   : std_logic_vector(7 downto 0) := X"44";
    constant COMMAND_PROG_SECURITY_REGISTERS    : std_logic_vector(7 downto 0) := X"42";
    constant COMMAND_READ_SECURITY_REGISTERS    : std_logic_vector(7 downto 0) := X"42";

   function link_36_to_std (
        i : in  integer--;
    ) return std_logic_vector;

end package;

package body mudaq is

    function link_36_to_std (
        i : in  integer--;
    ) return std_logic_vector is
    begin
        case i is
        when  0 => return "000000";
        when  1 => return "000001";
        when  2 => return "000010";
        when  3 => return "000011";
        when  4 => return "000100";
        when  5 => return "000101";
        when  6 => return "000110";
        when  7 => return "000111";
        when  8 => return "001000";
        when  9 => return "001001";
        when 10 => return "001010";
        when 11 => return "001011";
        when 12 => return "001100";
        when 13 => return "001101";
        when 14 => return "001110";
        when 15 => return "001111";
        when 16 => return "010000";
        when 17 => return "010001";
        when 18 => return "010010";
        when 19 => return "010011";
        when 20 => return "010100";
        when 21 => return "010101";
        when 22 => return "010110";
        when 23 => return "010111";
        when 24 => return "011000";
        when 25 => return "011001";
        when 26 => return "011010";
        when 27 => return "011011";
        when 28 => return "011100";
        when 29 => return "011101";
        when 30 => return "011110";
        when 31 => return "011111";
        when 32 => return "100000";
        when 33 => return "100001";
        when 34 => return "100010";
        when 35 => return "100011";
        when others =>
            return "111111";
        end case;
    end function;

end package body;
