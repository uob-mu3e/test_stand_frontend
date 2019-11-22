-- Basic constants for DAQ communication
-- K. Briggl, April 2019 : stripped from mupix8_daq repository / mupix_constants.vhd
-- last change: S. Dittmeier, 22.11.2019 (dittmeier@physi.uni-heidelberg.de)

library ieee;
use ieee.std_logic_1164.all;

package daq_constants is

-- multi-purpose types
subtype links_reg32 is std_logic_vector(31 downto 0);	-- same as reg32, used in MUTRIG setup with only one link?
subtype reg32 is std_logic_vector(31 downto 0);
constant NREGISTERS : integer := 64;
type reg32array is array (NREGISTERS-1 downto 0) of reg32;
type reg32array_t is array (natural range <>) of reg32;

subtype reg64 is std_logic_vector(63 downto 0);
type reg64array_t is array (natural range <>) of std_logic_vector(63 downto 0);

subtype byte_t is std_logic_vector(7 downto 0);
type bytearray_t is array (natural range <>)  of byte_t;

subtype REG64_TOP_RANGE is integer range 63 downto 32;
subtype REG64_BOTTOM_RANGE is integer range 31 downto 0;
-- general FEB constants
constant NLVDS				: integer := 32;	-- number of total links available
constant NINPUTS_BANK_A	: integer := 16;	-- number of links available on bank A (dividing LVDS banks into physical regions)
constant NINPUTS_BANK_B	: integer := 16;	-- number of links available on bank B (dividing LVDS banks into physical regions)
-- this should be equal to log2(NLVDS)
constant NLVDSLOG			: integer := 5;

-- two types below may be removed
-- types
type fifo_init_array is array (NLVDS-1 downto 0) of std_logic_vector(1 downto 0);
type fifo_usedw_array is array(NLVDS-1 downto 0) of std_logic_vector(3 downto 0);

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
constant k28_0 : std_logic_vector(7 downto 0) := X"1C"; -- used in MuPix
constant k28_1 : std_logic_vector(7 downto 0) := X"3C"; -- used in data alignment (transceiver)
constant k28_2 : std_logic_vector(7 downto 0) := X"5C";
constant k28_3 : std_logic_vector(7 downto 0) := X"7C";
constant k28_4 : std_logic_vector(7 downto 0) := X"9C";
constant k28_5 : std_logic_vector(7 downto 0) := X"BC"; -- used in MuPix
constant k28_6 : std_logic_vector(7 downto 0) := X"DC";
constant k28_7 : std_logic_vector(7 downto 0) := X"FC"; -- not used, comma symbol with harder constraints!
constant k23_7 : std_logic_vector(7 downto 0) := X"F7"; -- used as "empty" data (transceiver)
constant k27_7 : std_logic_vector(7 downto 0) := X"FB";
constant k29_7 : std_logic_vector(7 downto 0) := X"FD";
constant k30_7 : std_logic_vector(7 downto 0) := X"FE";


-- mscb addressing (for networks with 8bit and 16bit addresses, we will use 16 ?)
constant MSCB_CMD_ADDR_NODE16   : std_logic_vector(7 downto 0) := X"0A";
constant MSCB_CMD_ADDR_NODE8    : std_logic_vector(7 downto 0) := X"09";
constant MSCB_CMD_ADDR_GRP8     : std_logic_vector(7 downto 0) := X"11"; -- group addressing
constant MSCB_CMD_ADDR_GRP16    : std_logic_vector(7 downto 0) := X"12";
constant MSCB_CMD_ADDR_BC       : std_logic_vector(7 downto 0) := X"10"; --broadcast
constant MSCB_CMD_PING8         : std_logic_vector(7 downto 0) := X"19";
constant MSCB_CMD_PING16        : std_logic_vector(7 downto 0) := X"1A";

constant run_prep_acknowledge:          std_logic_vector(31 downto 0)	:= x"000001fe";
constant run_prep_acknowledge_datak:    std_logic_vector(3 downto 0) 	:= "0001";
constant RUN_END:                       std_logic_vector(31 downto 0)	:= x"000002fe";
constant RUN_END_DATAK:                 std_logic_vector(3 downto 0)	:= "0001";
constant MERGER_TIMEOUT:                std_logic_vector(31 downto 0)	:= x"000003fe";
constant MERGER_TIMEOUT_DATAK:          std_logic_vector(3 downto 0)	:= "0001";
    

end package;
