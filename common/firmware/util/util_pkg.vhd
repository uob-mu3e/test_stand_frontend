--
-- author : Alexandr Kozlinskiy
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use std.textio.all;
use ieee.std_logic_textio.all;

package mupix is
    --! MuPix
    -----------------------------------------------------------------
    -- Things to clean up with the generics
    -----------------------------------------------------------------
    constant NINPUTS                :  integer := 36;
    constant NSORTERINPUTS          :  integer :=  1;
    constant NCHIPS                 :  integer := 12;

    -----------------------------------------------------------------
    -- conflicts between detectorfpga_constants and mupix_constants (to be checked & tested)
    -----------------------------------------------------------------

    constant HITSIZE                :  integer := 32;

    constant TIMESTAMPSIZE          :  integer := 11;

    subtype TSRANGE                 is integer range TIMESTAMPSIZE-1 downto 0;

    constant COARSECOUNTERSIZE      :  integer := 32;

    subtype  COLRANGE               is integer range 23 downto 16;
    subtype  ROWRANGE               is integer range 15 downto 8;

    constant CHIPRANGE              :  integer := 3;

    -----------------------------------------------------------
    -----------------------------------------------------------

    constant BINCOUNTERSIZE         :  integer := 24;
    constant CHARGESIZE_MP10        :  integer := 5;
    constant SLOWTIMESTAMPSIZE      :  integer := 10;

    constant NOTSHITSIZE            :  integer := HITSIZE -TIMESTAMPSIZE;--HITSIZE -TIMESTAMPSIZE-1;
    subtype SLOWTSRANGE             is integer range TIMESTAMPSIZE-1 downto 1;
    subtype NOTSRANGE               is integer range HITSIZE-1 downto TIMESTAMPSIZE;--TIMESTAMPSIZE+1;

    constant HITSORTERBINBITS       :  integer := 4;
    constant H                      :  integer := HITSORTERBINBITS;
    constant HITSORTERADDRSIZE      :  integer := TIMESTAMPSIZE + HITSORTERBINBITS;

    constant BITSPERTSBLOCK         :  integer := 4;
    subtype TSBLOCKRANGE            is integer range TIMESTAMPSIZE-1 downto BITSPERTSBLOCK;
    subtype SLOWTSNONBLOCKRANGE     is integer range BITSPERTSBLOCK-2 downto 0;

    constant COMMANDBITS            :  integer := 20;

    constant COUNTERMEMADDRSIZE     :  integer := 8;
    constant NMEMS                  :  integer := 2**(TIMESTAMPSIZE-COUNTERMEMADDRSIZE-1); -- -1 due to even odd in single memory
    constant COUNTERMEMDATASIZE     :  integer := 10;
    subtype COUNTERMEMSELRANGE      is integer range TIMESTAMPSIZE-1 downto COUNTERMEMADDRSIZE+1;
    subtype SLOWTSCOUNTERMEMSELRANGE is integer range TIMESTAMPSIZE-2 downto COUNTERMEMADDRSIZE;
    subtype COUNTERMEMADDRRANGE     is integer range COUNTERMEMADDRSIZE downto 1;
    subtype SLOWCOUNTERMEMADDRRANGE is integer range COUNTERMEMADDRSIZE-1 downto 0;

    -- Bit positions in the counter fifo of the sorter
    subtype EVENCOUNTERRANGE        is integer range 2*NCHIPS*HITSORTERBINBITS-1 downto 0;
    constant EVENOVERFLOWBIT        :  integer := 2*NCHIPS*HITSORTERBINBITS;
    constant HASEVENBIT             :  integer := 2*NCHIPS*HITSORTERBINBITS+1;
    subtype ODDCOUNTERRANGE         is integer range 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT downto HASEVENBIT+1;
    constant ODDOVERFLOWBIT         :  integer := 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+1;
    constant HASODDBIT              :  integer := 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+2;
    subtype TSINFIFORANGE           is integer range 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+SLOWTIMESTAMPSIZE+2 downto 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+3;
    subtype TSBLOCKINFIFORANGE      is integer range TSINFIFORANGE'left downto TSINFIFORANGE'left-BITSPERTSBLOCK+1;
    subtype TSINBLOCKINFIFORANGE    is integer range TSINFIFORANGE'right+BITSPERTSBLOCK-2  downto TSINFIFORANGE'right;

    type ts_array_t                 is array (natural range <>) of std_logic_vector(10 downto 0);
    type row_array_t                is array (natural range <>) of std_logic_vector(7 downto 0);
    type col_array_t                is array (natural range <>) of std_logic_vector(7 downto 0);
    type ch_ID_array_t              is array (natural range <>) of std_logic_vector(5 downto 0);
    type tot_array_t                is array (natural range <>) of std_logic_vector(5 downto 0);

    subtype hit_t is                std_logic_vector(HITSIZE-1 downto 0);
    subtype cnt_t is                std_logic_vector(COARSECOUNTERSIZE-1  downto 0);
    subtype ts_t is                 std_logic_vector(TSRANGE);
    subtype slowts_t                is std_logic_vector(SLOWTIMESTAMPSIZE-1 downto 0);
    subtype nots_t                  is std_logic_vector(NOTSHITSIZE-1 downto 0);
    subtype addr_t                  is std_logic_vector(HITSORTERADDRSIZE-1 downto 0);
    subtype counter_t               is std_logic_vector(HITSORTERBINBITS-1 downto 0);

    constant counter1               :  counter_t := (others => '1');

    type wide_hit_array             is array (NINPUTS-1 downto 0) of hit_t;
    type hit_array                  is array (NCHIPS-1 downto 0) of hit_t;

    type wide_cnt_array             is array (NINPUTS-1 downto 0) of cnt_t;
    type cnt_array                  is array (NCHIPS-1 downto 0) of cnt_t;

    type ts_array                   is array (NCHIPS-1 downto 0) of ts_t;
    type slowts_array               is array (NCHIPS-1 downto 0) of slowts_t;

    type nots_hit_array             is array (NCHIPS-1 downto 0) of nots_t;
    type addr_array                 is array (NCHIPS-1 downto 0) of addr_t;

    type counter_chips              is array (NCHIPS-1 downto 0) of counter_t;
    subtype counter2_chips          is std_logic_vector(2*NCHIPS*HITSORTERBINBITS-1 downto 0);

    type hitcounter_sum3_type is array (NCHIPS/3-1 downto 0) of integer;

    subtype chip_bits_t             is std_logic_vector(NCHIPS-1 downto 0);

    subtype muxhit_t                is std_logic_vector(HITSIZE+1 downto 0);
    type muxhit_array               is array ((NINPUTS/4) downto 0) of muxhit_t;

    subtype byte_t                  is std_logic_vector(7 downto 0);
    type inbyte_array               is array (NINPUTS-1 downto 0) of byte_t;

    type state_type                 is (INIT, START, PRECOUNT, COUNT);

    subtype block_t                 is std_logic_vector(TSBLOCKRANGE);

    subtype command_t               is std_logic_vector(COMMANDBITS-1 downto 0);
    constant COMMAND_HEADER1        :  command_t := X"80000";
    constant COMMAND_HEADER2        :  command_t := X"90000";
    constant COMMAND_SUBHEADER      :  command_t := X"C0000";
    constant COMMAND_FOOTER         :  command_t := X"E0000";

    subtype doublecounter_t         is std_logic_vector(COUNTERMEMDATASIZE-1 downto 0);
    type doublecounter_array        is array (NMEMS-1 downto 0) of doublecounter_t;
    type doublecounter_chiparray    is array (NCHIPS-1 downto 0) of doublecounter_t;
    type alldoublecounter_array     is array (NCHIPS-1 downto 0) of doublecounter_array;

    subtype counteraddr_t           is std_logic_vector(COUNTERMEMADDRSIZE-1 downto 0);
    type counteraddr_array          is array (NMEMS-1 downto 0) of counteraddr_t;
    type counteraddr_chiparray      is array (NCHIPS-1 downto 0) of counteraddr_t;
    type allcounteraddr_array       is array (NCHIPS-1 downto 0) of counteraddr_array;

    type counterwren_array          is array (NMEMS-1 downto 0) of std_logic;
    type allcounterwren_array       is array (NCHIPS-1 downto 0) of counterwren_array;
    subtype countermemsel_t         is std_logic_vector(COUNTERMEMADDRRANGE);
    type reg_array                  is array (NCHIPS-1 downto 0) of work.util.reg32;
end package;



library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use std.textio.all;
use ieee.std_logic_textio.all;

package util is


    --! basic array types    
    subtype slv2_t is std_logic_vector(1 downto 0);
    type slv2_array_t is array ( natural range <> ) of slv2_t;
    subtype slv4_t is std_logic_vector(3 downto 0);
    type slv4_array_t is array ( natural range <> ) of slv4_t;
    subtype slv6_t is std_logic_vector(5 downto 0);
    type slv6_array_t is array ( natural range <> ) of slv6_t;
    subtype slv8_t is std_logic_vector(7 downto 0);
    type slv8_array_t is array ( natural range <> ) of slv8_t;
    subtype slv16_t is std_logic_vector(15 downto 0);
    type slv16_array_t is array ( natural range <> ) of slv16_t;
    subtype slv32_t is std_logic_vector(31 downto 0);
    type slv32_array_t is array ( natural range <> ) of slv32_t;
    subtype slv37_t is std_logic_vector(36 downto 0);
    type slv37_array_t is array ( natural range <> ) of slv37_t;
    subtype slv38_t is std_logic_vector(37 downto 0);
    type slv38_array_t is array ( natural range <> ) of slv38_t;
    subtype slv64_t is std_logic_vector(63 downto 0);
    type slv64_array_t is array ( natural range <> ) of slv64_t;
    subtype slv66_t is std_logic_vector(65 downto 0);
    type slv66_array_t is array ( natural range <> ) of slv66_t;
    subtype slv76_t is std_logic_vector(75 downto 0);
    type slv76_array_t is array ( natural range <> ) of slv76_t;
    subtype slv78_t is std_logic_vector(77 downto 0);
    type slv78_array_t is array ( natural range <> ) of slv78_t;
    subtype slv152_t is std_logic_vector(151 downto 0);
    type slv152_array_t is array ( natural range <> ) of slv152_t;
    subtype slv256_t is std_logic_vector(255 downto 0);
    type slv256_array_t is array ( natural range <> ) of slv256_t;

    type natural_array_t is array(integer range<>) of natural;

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
    subtype dataplusts_type is std_logic_vector(271 downto 0);
    type offset is array(natural range <>) of integer range 64 downto 0;
    subtype tsrange_type is std_logic_vector(15 downto 0);
    subtype tsupper is natural range 15 downto 8;
    subtype tslower is natural range 7 downto 0;
    constant tsone : tsrange_type := (others => '1');
    constant tszero : tsrange_type := (others => '0');


    --! data path swb types
    constant tree_padding : std_logic_vector(37 downto 0) := "11" & x"FFFFFFFFF";
    constant tree_paddingk : std_logic_vector(37 downto 0) := "11" & x"EEEEEEEEE";
    constant tree_zero : std_logic_vector(37 downto 0) := "00" & x"000000000";
    constant pre_marker : std_logic_vector(5 downto 0) := "110000";
    constant sh_marker : std_logic_vector(5 downto 0)  := "110001";
    constant tr_marker : std_logic_vector(5 downto 0)  := "110010";
    constant ts1_marker : std_logic_vector(5 downto 0) := "110011";
    constant ts2_marker : std_logic_vector(5 downto 0) := "110100";
    constant err_marker : std_logic_vector(5 downto 0) := "110101";


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


    --! PCIe types
    subtype reg32 is std_logic_vector(31 downto 0);
    constant NREGISTERS :  integer := 64;
    type reg32array is array (NREGISTERS-1 downto 0) of reg32;


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
    constant FEBSPI_ADDR_WRITENABLE         : std_logic_vector(6 downto 0)      := "0000001";
    constant FEBSPI_PATTERN_WRITENABLE      : std_logic_vector(7 downto 0)      := X"A3";
    constant FEBSPI_ADDR_STATUS             : std_logic_vector(6 downto 0)      := "0000010";
    constant FEBSPI_ADDR_CONTROL            : std_logic_vector(6 downto 0)      := "0000011";
    constant FEBSPI_ADDR_RESET              : std_logic_vector(6 downto 0)      := "0000100";
    constant FEBSPI_ADDR_PROGRAMMING_STATUS : std_logic_vector(6 downto 0)      := "0010000";
    constant FEBSPI_ADDR_PROGRAMMING_COUNT  : std_logic_vector(6 downto 0)      := "0010001";
    constant FEBSPI_ADDR_PROGRAMMING_CTRL   : std_logic_vector(6 downto 0)      := "0010010";
    constant FEBSPI_ADDR_PROGRAMMING_ADDR   : std_logic_vector(6 downto 0)      := "0010011";
    constant FEBSPI_ADDR_PROGRAMMING_WFIFO  : std_logic_vector(6 downto 0)      := "0010100";
    constant FEBSPI_ADDR_PROGRAMMING_RFIFO  : std_logic_vector(6 downto 0)      := "0010101";
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


    --! MuTrig
    CONSTANT C_HEAD_ID : std_logic_vector(7 downto 0) := "00011100";        --K28.0, 0x1C, 10 BIT: 1101000011
    CONSTANT C_TRAIL_ID : std_logic_vector(7 downto 0) := "10011100";       --K28.4, 0x9C, 10 BIT: 1011000011
    CONSTANT C_COMMA : std_logic_vector(7 downto 0) := "10111100";          --K28.5, 0xBC, 10 BIT: 1010000011, Sync character
    CONSTANT C_CLOCK_CORR_ID : std_logic_vector(7 downto 0) := "01111100";      --K28.3, 0x7C, 10 BIT: 0011000011, Clock Correction

    CONSTANT C_HEADER : std_logic_vector(7 downto 0) := C_HEAD_ID;
    CONSTANT C_TRAILER : std_logic_vector(7 downto 0) := C_TRAIL_ID;

    type mutrig_evtdata_array_t is array (natural range <>) of std_logic_vector(55 downto 0);   
    --spi pattern
    constant MUTRIG1_SPI_WORDS     : integer := 74;
    constant MUTRIG1_SPI_FIRST_MSB : integer := 21;
    constant STIC3_SPI_WORDS     : integer := 146;
    constant STIC3_SPI_FIRST_MSB : integer := 16;


    type avalon_t is record
        address         :   std_logic_vector(31 downto 0);
        read            :   std_logic;
        readdata        :   std_logic_vector(31 downto 0);
        write           :   std_logic;
        writedata       :   std_logic_vector(31 downto 0);
        waitrequest     :   std_logic;
        readdatavalid   :   std_logic;
    end record;
    type avalon_array_t is array(natural range <>) of avalon_t;



    -- avalon memory mapped interface
    type avmm_t is record
        address         :   std_logic_vector(31 downto 0);
        read            :   std_logic;
        readdata        :   std_logic_vector(31 downto 0);
        write           :   std_logic;
        writedata       :   std_logic_vector(31 downto 0);
        waitrequest     :   std_logic;
        readdatavalid   :   std_logic;
    end record;
    type avmm_array_t is array(natural range <>) of avmm_t;

    type rw_t is record
        addr            :   std_logic_vector(31 downto 0);
        re              :   std_logic; -- read enable
        rvalid          :   std_logic; -- read valid
        rdata           :   std_logic_vector(31 downto 0);
        we              :   std_logic; -- write enable
        wdata           :   std_logic_vector(31 downto 0);
    end record;



    -- Greatest Common Divisor
    function gcd (
        p, q : positive--;
    ) return positive;

    function max (
        l, r : integer--;
    ) return integer;

    function vector_width (
        v : natural--;
    ) return positive;

    function bin2gray (
        v : std_logic_vector--;
    ) return std_logic_vector;

    function gray2bin (
        v : std_logic_vector--;
    ) return std_logic_vector;

    function gray_inc (
        v : std_logic_vector--;
    ) return std_logic_vector;

    function shift_right (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector;

    function shift_left (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector;

    function resize (
        v : std_logic_vector;
        n : positive--;
    ) return std_logic_vector;

    function and_reduce (
        v : std_logic_vector--;
    ) return std_logic;

    function or_reduce (
        v : std_logic_vector--;
    ) return std_logic;

    function xor_reduce (
        v : std_logic_vector--;
    ) return std_logic;

    function to_std_logic (
        b : in boolean--;
    ) return std_logic;

    function reverse (
        v : std_logic_vector--;
    ) return std_logic_vector;

    procedure char_to_hex (
        c : in character;
        v : out std_logic_vector(3 downto 0);
        good : out boolean--;
    );

    procedure string_to_hex (
        s : in string;
        v : out std_logic_vector;
        good : out boolean--;
    );

    procedure read_hex (
        l : inout line;
        value : out std_logic_vector;
        good : out boolean--;
    );

    function hex_to_ascii (
        h : in std_logic_vector--;
    ) return std_logic_vector;

    function link_36_to_std (
        i : in integer--;
    ) return std_logic_vector;


    -- LFSR 32
    -- src: http://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
    --
    -- taps: 31, 21, 1, 0
    function lfsr_32 (
        data : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector;

    -- CRC-32C (Castagnoli) 0x1.1EDC6F41
    -- src: http://www.easics.com/services/freesics/crctool.html
    -- polynomial: x^32 + x^28 + x^27 + x^26 + x^25 + x^23 + x^22 + x^20 + x^19 + x^18 + x^14 + x^13 + x^11 + x^10 + x^9 + x^8 + x^6 + 1
    -- data width: 32
    -- convention: the first serial bit is D[31] -- TODO: D[0]
    function crc32 (
        data : std_logic_vector(31 downto 0);
        crc  : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector;



    function count_bits_4 (
        data : std_logic_vector(3 downto 0)--;
    ) return natural;

    function count_bits_32 (
        data : std_logic_vector(31 downto 0)--;
    ) return natural;

    function count_bits (
        data : std_logic_vector--;
    ) return natural;

    impure
    function read_hex (
        fname : in string;
        N : in positive;
        W : in positive--;
    ) return std_logic_vector;

    function to_string (
        v : in std_logic--;
    ) return string;

    function to_string (
        v : in std_logic_vector--;
    ) return string;

    function to_string (
        v : in unsigned--;
    ) return string;

    function to_hstring (
        v : std_logic_vector--;
    ) return string;

    function to_hstring (
        v : unsigned--;
    ) return string;

    -- Select Graphic Rendition
    function sgr (
        n : natural--;
    ) return string;

end package;

package body util is

    function gcd (
        p, q : positive--;
    ) return positive is
        variable p_v : positive := p;
        variable q_v : positive := q;
    begin
        while ( p_v /= q_v ) loop
            if ( p_v > q_v ) then
                p_v := p_v - q_v;
            else
                q_v := q_v - p_v;
            end if;
        end loop;
        return p_v;
    end function;

    function max (
        l, r : integer
    ) return integer is
    begin
        if l > r then
            return l;
        else
            return r;
        end if;
    end function;

    function vector_width (
        v : natural--;
    ) return positive is
    begin
        if ( v = 0 or v = 1 ) then
            return 1;
        end if;
        return positive(ceil(log2(real(v))));
    end function;

    function bin2gray (
        v : std_logic_vector--;
    ) return std_logic_vector is
    begin
        return v xor shift_right(v, 1);
    end function;

    function gray2bin (
        v : std_logic_vector--;
    ) return std_logic_vector is
        variable b : std_logic := '0';
        variable r : std_logic_vector(v'range);
    begin
        for i in v'range loop
            b := b xor v(i);
            r(i) := b;
        end loop;
        return r;
    end function;

    function gray_inc (
        v : std_logic_vector--;
    ) return std_logic_vector is
        variable r : std_logic_vector(v'range) := (others => '0');
    begin
        r := gray2bin(v);
        r := std_logic_vector(unsigned(r) + 1);
        return bin2gray(r);
    end function;

    function shift_right (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector is
    begin
        return std_logic_vector(shift_right(unsigned(v), n));
    end function;

    function shift_left (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector is
    begin
        return std_logic_vector(shift_left(unsigned(v), n));
    end function;

    function resize (
        v : std_logic_vector;
        n : positive--;
    ) return std_logic_vector is
    begin
        return std_logic_vector(resize(unsigned(v), n));
    end function;

    function and_reduce (
        v : std_logic_vector--;
    ) return std_logic is
    begin
        return to_std_logic(v = (v'range => '1'));
    end function;

    function or_reduce (
        v : std_logic_vector--;
    ) return std_logic is
    begin
        return to_std_logic(v /= (v'range => '0'));
    end function;

    function xor_reduce (
        v : std_logic_vector--;
    ) return std_logic is
        alias a : std_logic_vector(v'length-1 downto 0) is v;
    begin
        if ( v'length = 0 ) then
            report "(xor_reduce) v'length = 0" severity failure;
            return 'X';
        end if;
        if ( a'length = 1 ) then
            return a(0);
        end if;
        return xor_reduce(a(a'length-1 downto a'length/2)) xor xor_reduce(a(a'length/2-1 downto 0));
    end function;

    function to_std_logic (
        b : in boolean--;
    ) return std_logic is
    begin
        if b then
            return '1';
        else
            return '0';
        end if;
    end function;

    function reverse (
        v : std_logic_vector--;
    ) return std_logic_vector is
        variable r : std_logic_vector(v'range);
        alias a : std_logic_vector(v'reverse_range) is v;
    begin
        for i in a'range loop
            r(i) := a(i);
        end loop;
        return r;
    end function;

    procedure char_to_hex (
        c : in character;
        v : out std_logic_vector(3 downto 0);
        good : out boolean--;
    ) is
    begin
        good := true;
        case c is
        when '0' => v := X"0";
        when '1' => v := X"1";
        when '2' => v := X"2";
        when '3' => v := X"3";
        when '4' => v := X"4";
        when '5' => v := X"5";
        when '6' => v := X"6";
        when '7' => v := X"7";
        when '8' => v := X"8";
        when '9' => v := X"9";

        when 'a' | 'A' => v := X"A";
        when 'b' | 'B' => v := X"B";
        when 'c' | 'C' => v := X"C";
        when 'd' | 'D' => v := X"D";
        when 'e' | 'E' => v := X"E";
        when 'f' | 'F' => v := X"F";

        when others =>
           report "(char_to_hex) invalid hex character '" & c & "'" severity failure;
           good := false;
           v := "XXXX";
        end case;
    end procedure;

    function hex_to_ascii (
        h : in  std_logic_vector--;
    ) return std_logic_vector is
    
    begin
        case h is
        when x"0" => return X"30";
        when x"1" => return X"31";
        when x"2" => return X"32";
        when x"3" => return X"33";
        when x"4" => return X"34";
        when x"5" => return X"35";
        when x"6" => return X"36";
        when x"7" => return X"37";
        when x"8" => return X"38";
        when x"9" => return X"39";
        when x"A" => return X"41";
        when x"B" => return X"42";
        when x"C" => return X"43";
        when x"D" => return X"44";
        when x"E" => return X"45";
        when x"F" => return X"46";

        when others =>
            return x"3F";
        end case;
    end function;

    procedure string_to_hex (
        s : in string;
        v : out std_logic_vector;
        good : out boolean--;
    ) is
        variable ok : boolean;
        variable good_i : boolean;
    begin
        good_i := true;
        for i in 0 to s'length-1 loop
            char_to_hex(s(s'right-i), v(3+4*i+v'right downto 4*i+v'right), ok);
            good_i := good_i and ok;
        end loop;
        good := good_i;
    end procedure;

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

    procedure read_hex (
        l : inout line;
        value : out std_logic_vector;
        good : out boolean--;
    ) is
        variable v : std_logic_vector(value'range);
        variable c : character;
        variable s : string(1 to value'length/4);
        variable ok : boolean;
    begin
        good := false;

        if value'length mod 4 /= 0 then
            report "(read_hex) value'length mod 4 /= 0" severity failure;
            return;
        end if;

        -- skip spaces
        loop
            read(l, c);
            exit when ((c /= ' ') and (c /= CR) and (c /= HT));
        end loop;

        -- skip comment
        if c = '#' then
            return;
        end if;

        s(1) := c;
        read(L, s(2 to s'right), ok);
        if not ok then
            return;
        end if;

        string_to_hex(s, v, ok);
        if not ok then
            return;
        end if;

        value := v;
        good := true;
    end procedure;

    impure
    function read_hex (
        fname : in string;
        N : in positive;
        W : in positive--;
    ) return std_logic_vector is
        variable data : std_logic_vector(N*W-1 downto 0);
        variable data_i : std_logic_vector(W-1 downto 0);
        variable i : integer := 0;
        file f : text;
        variable fs : file_open_status;
        variable l : line;
        variable c : character;
        variable s : string(1 to W/4);
        variable ok : boolean;
    begin
        if fname'length = 0 then
            return data;
        end if;

        file_open(fs, f, fname, READ_MODE);
        assert ( fs = open_ok ) report "(read_hex) file_open_status = '" & FILE_OPEN_STATUS'image(fs) & "'" severity failure;

        while ( not endfile(f) ) loop
            readline(f, l);
            read(l, c, ok);
            next when ( not ok or c = '#' );
            s(1) := c;
            read(l, s(2 to s'right), ok);
            next when ( not ok );
            work.util.string_to_hex(s, data_i, ok);
            next when ( not ok );
            data(W-1+i*W downto i*W) := data_i;
            i := i + 1;
        end loop;

        file_close(f);
        return data;
    end function;



    function lfsr_32(
        data : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector is
    begin
        return data(30 downto 0) &
              (data(31) xor data(21) xor data(1) xor data(0));
    end function;

    function crc32 (
        data : std_logic_vector(31 downto 0);
        crc  : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector is
        variable d      : std_logic_vector(31 downto 0);
        variable c      : std_logic_vector(31 downto 0);
        variable newcrc : std_logic_vector(31 downto 0);
    begin
        d := data;
        c := crc;

        newcrc(0) := d(31) xor d(30) xor d(28) xor d(27) xor d(26) xor d(25) xor d(23) xor d(21) xor d(18) xor d(17) xor d(16) xor d(12) xor d(9) xor d(8) xor d(7) xor d(6) xor d(5) xor d(4) xor d(0) xor c(0) xor c(4) xor c(5) xor c(6) xor c(7) xor c(8) xor c(9) xor c(12) xor c(16) xor c(17) xor c(18) xor c(21) xor c(23) xor c(25) xor c(26) xor c(27) xor c(28) xor c(30) xor c(31);
        newcrc(1) := d(31) xor d(29) xor d(28) xor d(27) xor d(26) xor d(24) xor d(22) xor d(19) xor d(18) xor d(17) xor d(13) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(5) xor d(1) xor c(1) xor c(5) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(13) xor c(17) xor c(18) xor c(19) xor c(22) xor c(24) xor c(26) xor c(27) xor c(28) xor c(29) xor c(31);
        newcrc(2) := d(30) xor d(29) xor d(28) xor d(27) xor d(25) xor d(23) xor d(20) xor d(19) xor d(18) xor d(14) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(2) xor c(2) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(14) xor c(18) xor c(19) xor c(20) xor c(23) xor c(25) xor c(27) xor c(28) xor c(29) xor c(30);
        newcrc(3) := d(31) xor d(30) xor d(29) xor d(28) xor d(26) xor d(24) xor d(21) xor d(20) xor d(19) xor d(15) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(3) xor c(3) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(15) xor c(19) xor c(20) xor c(21) xor c(24) xor c(26) xor c(28) xor c(29) xor c(30) xor c(31);
        newcrc(4) := d(31) xor d(30) xor d(29) xor d(27) xor d(25) xor d(22) xor d(21) xor d(20) xor d(16) xor d(13) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(4) xor c(4) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(13) xor c(16) xor c(20) xor c(21) xor c(22) xor c(25) xor c(27) xor c(29) xor c(30) xor c(31);
        newcrc(5) := d(31) xor d(30) xor d(28) xor d(26) xor d(23) xor d(22) xor d(21) xor d(17) xor d(14) xor d(13) xor d(12) xor d(11) xor d(10) xor d(9) xor d(5) xor c(5) xor c(9) xor c(10) xor c(11) xor c(12) xor c(13) xor c(14) xor c(17) xor c(21) xor c(22) xor c(23) xor c(26) xor c(28) xor c(30) xor c(31);
        newcrc(6) := d(30) xor d(29) xor d(28) xor d(26) xor d(25) xor d(24) xor d(22) xor d(21) xor d(17) xor d(16) xor d(15) xor d(14) xor d(13) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(5) xor d(4) xor d(0) xor c(0) xor c(4) xor c(5) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(13) xor c(14) xor c(15) xor c(16) xor c(17) xor c(21) xor c(22) xor c(24) xor c(25) xor c(26) xor c(28) xor c(29) xor c(30);
        newcrc(7) := d(31) xor d(30) xor d(29) xor d(27) xor d(26) xor d(25) xor d(23) xor d(22) xor d(18) xor d(17) xor d(16) xor d(15) xor d(14) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(6) xor d(5) xor d(1) xor c(1) xor c(5) xor c(6) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(14) xor c(15) xor c(16) xor c(17) xor c(18) xor c(22) xor c(23) xor c(25) xor c(26) xor c(27) xor c(29) xor c(30) xor c(31);
        newcrc(8) := d(25) xor d(24) xor d(21) xor d(19) xor d(15) xor d(13) xor d(11) xor d(10) xor d(8) xor d(5) xor d(4) xor d(2) xor d(0) xor c(0) xor c(2) xor c(4) xor c(5) xor c(8) xor c(10) xor c(11) xor c(13) xor c(15) xor c(19) xor c(21) xor c(24) xor c(25);
        newcrc(9) := d(31) xor d(30) xor d(28) xor d(27) xor d(23) xor d(22) xor d(21) xor d(20) xor d(18) xor d(17) xor d(14) xor d(11) xor d(8) xor d(7) xor d(4) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(4) xor c(7) xor c(8) xor c(11) xor c(14) xor c(17) xor c(18) xor c(20) xor c(21) xor c(22) xor c(23) xor c(27) xor c(28) xor c(30) xor c(31);
        newcrc(10) := d(30) xor d(29) xor d(27) xor d(26) xor d(25) xor d(24) xor d(22) xor d(19) xor d(17) xor d(16) xor d(15) xor d(7) xor d(6) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(6) xor c(7) xor c(15) xor c(16) xor c(17) xor c(19) xor c(22) xor c(24) xor c(25) xor c(26) xor c(27) xor c(29) xor c(30);
        newcrc(11) := d(21) xor d(20) xor d(12) xor d(9) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(9) xor c(12) xor c(20) xor c(21);
        newcrc(12) := d(22) xor d(21) xor d(13) xor d(10) xor d(7) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(7) xor c(10) xor c(13) xor c(21) xor c(22);
        newcrc(13) := d(31) xor d(30) xor d(28) xor d(27) xor d(26) xor d(25) xor d(22) xor d(21) xor d(18) xor d(17) xor d(16) xor d(14) xor d(12) xor d(11) xor d(9) xor d(3) xor d(2) xor d(0) xor c(0) xor c(2) xor c(3) xor c(9) xor c(11) xor c(12) xor c(14) xor c(16) xor c(17) xor c(18) xor c(21) xor c(22) xor c(25) xor c(26) xor c(27) xor c(28) xor c(30) xor c(31);
        newcrc(14) := d(30) xor d(29) xor d(25) xor d(22) xor d(21) xor d(19) xor d(16) xor d(15) xor d(13) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(5) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(5) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(13) xor c(15) xor c(16) xor c(19) xor c(21) xor c(22) xor c(25) xor c(29) xor c(30);
        newcrc(15) := d(31) xor d(30) xor d(26) xor d(23) xor d(22) xor d(20) xor d(17) xor d(16) xor d(14) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(4) xor d(2) xor d(1) xor c(1) xor c(2) xor c(4) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(14) xor c(16) xor c(17) xor c(20) xor c(22) xor c(23) xor c(26) xor c(30) xor c(31);
        newcrc(16) := d(31) xor d(27) xor d(24) xor d(23) xor d(21) xor d(18) xor d(17) xor d(15) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(5) xor d(3) xor d(2) xor c(2) xor c(3) xor c(5) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(15) xor c(17) xor c(18) xor c(21) xor c(23) xor c(24) xor c(27) xor c(31);
        newcrc(17) := d(28) xor d(25) xor d(24) xor d(22) xor d(19) xor d(18) xor d(16) xor d(13) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(6) xor d(4) xor d(3) xor c(3) xor c(4) xor c(6) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(13) xor c(16) xor c(18) xor c(19) xor c(22) xor c(24) xor c(25) xor c(28);
        newcrc(18) := d(31) xor d(30) xor d(29) xor d(28) xor d(27) xor d(21) xor d(20) xor d(19) xor d(18) xor d(16) xor d(14) xor d(13) xor d(11) xor d(10) xor d(8) xor d(6) xor d(0) xor c(0) xor c(6) xor c(8) xor c(10) xor c(11) xor c(13) xor c(14) xor c(16) xor c(18) xor c(19) xor c(20) xor c(21) xor c(27) xor c(28) xor c(29) xor c(30) xor c(31);
        newcrc(19) := d(29) xor d(27) xor d(26) xor d(25) xor d(23) xor d(22) xor d(20) xor d(19) xor d(18) xor d(16) xor d(15) xor d(14) xor d(11) xor d(8) xor d(6) xor d(5) xor d(4) xor d(1) xor d(0) xor c(0) xor c(1) xor c(4) xor c(5) xor c(6) xor c(8) xor c(11) xor c(14) xor c(15) xor c(16) xor c(18) xor c(19) xor c(20) xor c(22) xor c(23) xor c(25) xor c(26) xor c(27) xor c(29);
        newcrc(20) := d(31) xor d(25) xor d(24) xor d(20) xor d(19) xor d(18) xor d(15) xor d(8) xor d(4) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(4) xor c(8) xor c(15) xor c(18) xor c(19) xor c(20) xor c(24) xor c(25) xor c(31);
        newcrc(21) := d(26) xor d(25) xor d(21) xor d(20) xor d(19) xor d(16) xor d(9) xor d(5) xor d(3) xor d(2) xor d(1) xor c(1) xor c(2) xor c(3) xor c(5) xor c(9) xor c(16) xor c(19) xor c(20) xor c(21) xor c(25) xor c(26);
        newcrc(22) := d(31) xor d(30) xor d(28) xor d(25) xor d(23) xor d(22) xor d(20) xor d(18) xor d(16) xor d(12) xor d(10) xor d(9) xor d(8) xor d(7) xor d(5) xor d(3) xor d(2) xor d(0) xor c(0) xor c(2) xor c(3) xor c(5) xor c(7) xor c(8) xor c(9) xor c(10) xor c(12) xor c(16) xor c(18) xor c(20) xor c(22) xor c(23) xor c(25) xor c(28) xor c(30) xor c(31);
        newcrc(23) := d(30) xor d(29) xor d(28) xor d(27) xor d(25) xor d(24) xor d(19) xor d(18) xor d(16) xor d(13) xor d(12) xor d(11) xor d(10) xor d(7) xor d(5) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(5) xor c(7) xor c(10) xor c(11) xor c(12) xor c(13) xor c(16) xor c(18) xor c(19) xor c(24) xor c(25) xor c(27) xor c(28) xor c(29) xor c(30);
        newcrc(24) := d(31) xor d(30) xor d(29) xor d(28) xor d(26) xor d(25) xor d(20) xor d(19) xor d(17) xor d(14) xor d(13) xor d(12) xor d(11) xor d(8) xor d(6) xor d(4) xor d(2) xor d(1) xor c(1) xor c(2) xor c(4) xor c(6) xor c(8) xor c(11) xor c(12) xor c(13) xor c(14) xor c(17) xor c(19) xor c(20) xor c(25) xor c(26) xor c(28) xor c(29) xor c(30) xor c(31);
        newcrc(25) := d(29) xor d(28) xor d(25) xor d(23) xor d(20) xor d(17) xor d(16) xor d(15) xor d(14) xor d(13) xor d(8) xor d(6) xor d(4) xor d(3) xor d(2) xor d(0) xor c(0) xor c(2) xor c(3) xor c(4) xor c(6) xor c(8) xor c(13) xor c(14) xor c(15) xor c(16) xor c(17) xor c(20) xor c(23) xor c(25) xor c(28) xor c(29);
        newcrc(26) := d(31) xor d(29) xor d(28) xor d(27) xor d(25) xor d(24) xor d(23) xor d(15) xor d(14) xor d(12) xor d(8) xor d(6) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(6) xor c(8) xor c(12) xor c(14) xor c(15) xor c(23) xor c(24) xor c(25) xor c(27) xor c(28) xor c(29) xor c(31);
        newcrc(27) := d(31) xor d(29) xor d(27) xor d(24) xor d(23) xor d(21) xor d(18) xor d(17) xor d(15) xor d(13) xor d(12) xor d(8) xor d(6) xor d(5) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(5) xor c(6) xor c(8) xor c(12) xor c(13) xor c(15) xor c(17) xor c(18) xor c(21) xor c(23) xor c(24) xor c(27) xor c(29) xor c(31);
        newcrc(28) := d(31) xor d(27) xor d(26) xor d(24) xor d(23) xor d(22) xor d(21) xor d(19) xor d(17) xor d(14) xor d(13) xor d(12) xor d(8) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(8) xor c(12) xor c(13) xor c(14) xor c(17) xor c(19) xor c(21) xor c(22) xor c(23) xor c(24) xor c(26) xor c(27) xor c(31);
        newcrc(29) := d(28) xor d(27) xor d(25) xor d(24) xor d(23) xor d(22) xor d(20) xor d(18) xor d(15) xor d(14) xor d(13) xor d(9) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(9) xor c(13) xor c(14) xor c(15) xor c(18) xor c(20) xor c(22) xor c(23) xor c(24) xor c(25) xor c(27) xor c(28);
        newcrc(30) := d(29) xor d(28) xor d(26) xor d(25) xor d(24) xor d(23) xor d(21) xor d(19) xor d(16) xor d(15) xor d(14) xor d(10) xor d(7) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(7) xor c(10) xor c(14) xor c(15) xor c(16) xor c(19) xor c(21) xor c(23) xor c(24) xor c(25) xor c(26) xor c(28) xor c(29);
        newcrc(31) := d(30) xor d(29) xor d(27) xor d(26) xor d(25) xor d(24) xor d(22) xor d(20) xor d(17) xor d(16) xor d(15) xor d(11) xor d(8) xor d(7) xor d(6) xor d(5) xor d(4) xor d(3) xor c(3) xor c(4) xor c(5) xor c(6) xor c(7) xor c(8) xor c(11) xor c(15) xor c(16) xor c(17) xor c(20) xor c(22) xor c(24) xor c(25) xor c(26) xor c(27) xor c(29) xor c(30);

        return newcrc;
    end function;



    function count_bits_4(
        data : std_logic_vector(3 downto 0)--;
    ) return natural is
    begin
        case data is
        when "0000" => return 0;
        when "0001" | "0010" | "0100" | "1000" => return 1;
        when "0111" | "1011" | "1101" | "1110" => return 3;
        when "1111" => return 4;
        when others => return 2;
        end case;
    end function;

    function count_bits_32(
        data : std_logic_vector(31 downto 0)--;
    ) return natural is
    begin
        return (
            (
                count_bits_4(data(31 downto 28)) +
                count_bits_4(data(27 downto 24))
            ) + (
                count_bits_4(data(23 downto 20)) +
                count_bits_4(data(19 downto 16))
            )
        ) + (
            (
                count_bits_4(data(15 downto 12)) +
                count_bits_4(data(11 downto  8))
            ) + (
                count_bits_4(data( 7 downto  4)) +
                count_bits_4(data( 3 downto  0))
            )
        );
    end function;

    function count_bits (
        data : std_logic_vector--;
    ) return natural is
        variable data_v : std_logic_vector(data'length-1 downto 0);
    begin
        data_v := data;
        if ( data_v'length > 1 ) then
            return count_bits(data_v(data_v'length-1 downto data_v'length/2)) + count_bits(data_v(data_v'length/2-1 downto 0));
        else
            return to_integer(unsigned(data_v));
        end if;
    end function;

    function to_string (
        v : in std_logic--;
    ) return string is
        variable s : string(1 to 1);
    begin
        s(1) := std_logic'image(v)(2);
        return s;
    end function;

    function to_string (
        v : in std_logic_vector--;
    ) return string is
        variable s : string(1 to v'length);
        variable j : integer := 1;
    begin
        for i in v'range loop
            s(j) := to_string(v(i))(1);
            j := j + 1;
        end loop;
        return s;
    end function;

    function to_string (
        v : in unsigned--;
    ) return string is
    begin
        return to_string(std_logic_vector(v));
    end function;

    function to_hstring (
        v : std_logic_vector--;
    ) return string is
        variable r : string(1 to (v'length + 3) / 4) := (others => 'X');
        variable u : unsigned(v'length+3 downto 0);
        constant lut : string(1 to 16) := "0123456789ABCDEF";
    begin
        u := resize(unsigned(v), u'length);
        for i in r'range loop
            next when ( is_x(std_logic_vector(u(4*i-1 downto 4*i-4))) );
            r(r'length-i+1) := lut(1 + to_integer(u(4*i-1 downto 4*i-4)));
        end loop;
        return r;
    end function;

    function to_hstring (
        v : unsigned--;
    ) return string is
    begin
        return to_hstring(std_logic_vector(v));
    end function;

    function sgr (
        n : natural--;
    ) return string is
    begin
        return ESC & "[" & natural'image(n) & "m";
    end function;

end package body;
