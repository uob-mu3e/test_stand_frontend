-- Basic constants for DAQ communication
-- K. Briggl, April 2019 : stripped from mupix8_daq repository / mupix_constants.vhd
-- last change: M. Mueller, 02.11.2020

library ieee;
use ieee.std_logic_1164.all;

package daq_constants is

-- multi-purpose types
subtype reg32                   is std_logic_vector(31 downto 0);
subtype reg16                   is std_logic_vector(15 downto 0);
subtype reg64                   is std_logic_vector(63 downto 0);
constant NREGISTERS             :  integer := 64;
type reg32array                 is array (NREGISTERS-1 downto 0) of reg32;
type reg32array_t               is array (natural range <>) of reg32;
type reg16array_t               is array (natural range <>) of reg16;
type reg32array_128             is array (128-1 downto 0) of reg32;
type reg64array_t               is array (natural range <>) of std_logic_vector(63 downto 0);

subtype byte_t                  is std_logic_vector(7 downto 0);
type bytearray_t                is array (natural range <>)  of byte_t;

subtype REG64_TOP_RANGE         is integer range 63 downto 32;
subtype REG64_BOTTOM_RANGE      is integer range 31 downto 0;
subtype REG62_TOP_RANGE         is integer range 61 downto 31;
subtype REG62_BOTTOM_RANGE      is integer range 30 downto 0;

type natural_array_t            is array(integer range<>) of natural;

type chips_reg32                is array (3 downto 0) of reg32;
type output_reg32               is array (14 downto 0) of reg32;


-- general FEB constants
constant NLVDS                  :  integer := 32;	-- number of total links available
constant NINPUTS_BANK_A         :  integer := 16;	-- number of links available on bank A (dividing LVDS banks into physical regions)
constant NINPUTS_BANK_B         :  integer := 16;	-- number of links available on bank B (dividing LVDS banks into physical regions)
-- this should be equal to log2(NLVDS)
constant NLVDSLOG               :  integer := 5;

-- type for run state
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



-- 8bit/10bit encoding
constant k28_0 : std_logic_vector(7 downto 0) := X"1C"; -- still used in MuPix ??
constant k28_1 : std_logic_vector(7 downto 0) := X"3C"; -- still used in data alignment (transceiver) ??
constant k28_2 : std_logic_vector(7 downto 0) := X"5C";
constant k28_3 : std_logic_vector(7 downto 0) := X"7C";
constant k28_4 : std_logic_vector(7 downto 0) := X"9C"; -- used as end of packet marker between FEB <--> SW board
constant k28_5 : std_logic_vector(7 downto 0) := X"BC"; -- still used in MuPix ???
constant k28_6 : std_logic_vector(7 downto 0) := X"DC";
constant k28_7 : std_logic_vector(7 downto 0) := X"FC"; -- not used, comma symbol with harder constraints!
constant k23_7 : std_logic_vector(7 downto 0) := X"F7"; -- still used as "empty" data (transceiver) ??
constant k27_7 : std_logic_vector(7 downto 0) := X"FB";
constant k29_7 : std_logic_vector(7 downto 0) := X"FD";
constant k30_7 : std_logic_vector(7 downto 0) := X"FE";


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
constant FEBSPI_ADDR_RESET              : std_logic_vector(6 downto 0)      := "0000011";
constant FEBSPI_ADDR_PROGRAMMING_STATUS : std_logic_vector(6 downto 0)      := "0010000";
constant FEBSPI_ADDR_PROGRAMMING_COUNT  : std_logic_vector(6 downto 0)      := "0010001";
constant FEBSPI_ADDR_PROGRAMMING_CTRL   : std_logic_vector(6 downto 0)      := "0010010";
constant FEBSPI_ADDR_PROGRAMMING_ADDR   : std_logic_vector(6 downto 0)      := "0010011";
constant FEBSPI_ADDR_PROGRAMMING_WFIFO  : std_logic_vector(6 downto 0)      := "0010100";
constant FEBSPI_ADDR_PROGRAMMING_RFIFO  : std_logic_vector(6 downto 0)      := "0010101";
constant FEBSPI_ADDR_ADCDATA            : std_logic_vector(6 downto 0)      := "0100000";
constant FEBSPI_ADDR_ADCCTRL            : std_logic_vector(6 downto 0)      := "0100001";

end package;
