---------------------------------------
--
-- On detector FPGA for layer 0 - constant library
-- Sebastian Dittmeier, June 2016
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;
use work.detectorfpga_constants.all;

package detectorfpga_types is

subtype hit_t is std_logic_vector(HITSIZE-1 downto 0);
subtype cnt_t is std_logic_vector(COARSECOUNTERSIZE-1  downto 0);
subtype ts_t is std_logic_vector(TSRANGE);
subtype slowts_t is std_logic_vector(SLOWTIMESTAMPSIZE-1 downto 0);
subtype nots_t is std_logic_vector(NOTSHITSIZE-1 downto 0);
subtype addr_t is std_logic_vector(HITSORTERADDRSIZE-1 downto 0);
subtype counter_t is std_logic_vector(HITSORTERBINBITS-1 downto 0);

constant counter1 : counter_t := (others => '1');

type wide_hit_array is array (NINPUTS-1 downto 0) of hit_t;
type hit_array is array (NCHIPS-1 downto 0) of hit_t;

type wide_cnt_array is array (NINPUTS-1 downto 0) of cnt_t;
type cnt_array is array (NCHIPS-1 downto 0) of cnt_t;

type ts_array is array (NCHIPS-1 downto 0) of ts_t;
type slowts_array is array (NCHIPS-1 downto 0) of slowts_t;

type nots_hit_array is array (NCHIPS-1 downto 0) of nots_t;
type addr_array is array (NCHIPS-1 downto 0) of addr_t;

subtype reg32 		is std_logic_vector(31 downto 0);
type reg_array is array (NCHIPS-1 downto 0) of reg32;

type counter_set is array (NTIMESTAMPS-1 downto 0) of counter_t;
type counter_array is array (NCHIPS-1 downto 0) of counter_set;
type counter_chips is array (NCHIPS-1 downto 0) of counter_t;
subtype counter2_chips is std_logic_vector(2*NCHIPS*HITSORTERBINBITS-1 downto 0);
--array (2*NCHIPS-1 downto 0) of counter_t;

type hitcounter_sum3_type is array (NCHIPS/3-1 downto 0) of integer;

subtype bit_set is std_logic_vector(NTIMESTAMPS-1 downto 0);
type bit_array is array (NCHIPS-1 downto 0) of bit_set;
subtype chip_bits_t is std_logic_vector(NCHIPS-1 downto 0);

subtype muxhit_t is std_logic_vector(HITSIZE+1 downto 0);
type muxhit_array is array ((NINPUTS/4) downto 0) of muxhit_t;

subtype byte_t is std_logic_vector(7 downto 0);
type inbyte_array is array (NINPUTS-1 downto 0) of byte_t;

type state_type is (INIT, START, PRECOUNT, COUNT);

subtype block_t	is std_logic_vector(TSBLOCKRANGE);
subtype blockbits_t is std_logic_vector(NUMTSBLOCKS-1 downto 0);
subtype inblockbits_t is std_logic_vector(TSPERBLOCK-1 downto 0);

subtype command_t is std_logic_vector(COMMANDBITS-1 downto 0);
constant COMMAND_HEADER1 : command_t := X"80000";
constant COMMAND_HEADER2 : command_t := X"90000";
constant COMMAND_SUBHEADER : command_t := X"C0000";
constant COMMAND_FOOTER : command_t := X"E0000";

subtype doublecounter_t is std_logic_vector(COUNTERMEMDATASIZE-1 downto 0);
type doublecounter_array is array (NMEMS-1 downto 0) of doublecounter_t;
type doublecounter_chiparray is array (NCHIPS-1 downto 0) of doublecounter_t;
type alldoublecounter_array is array (NCHIPS-1 downto 0) of doublecounter_array;

subtype counteraddr_t is std_logic_vector(COUNTERMEMADDRSIZE-1 downto 0);
type counteraddr_array is array (NMEMS-1 downto 0) of counteraddr_t;
type counteraddr_chiparray is array (NCHIPS-1 downto 0) of counteraddr_t;
type allcounteraddr_array is array (NCHIPS-1 downto 0) of counteraddr_array;

type counterwren_array is array (NMEMS-1 downto 0) of std_logic;
type allcounterwren_array is array (NCHIPS-1 downto 0) of counterwren_array;

subtype countermemsel_t is std_logic_vector(COUNTERMEMADDRRANGE);

-- for hitsorter...

subtype reg64 		is std_logic_vector(63 downto 0);
type chips_reg32	is array (3 downto 0) of reg32;
subtype REG64_TOP_RANGE is integer range 63 downto 32;
subtype REG64_BOTTOM_RANGE is integer range 31 downto 0;
subtype REG62_TOP_RANGE is integer range 61 downto 31;
subtype REG62_BOTTOM_RANGE is integer range 30 downto 0;
type output_reg32	 is array (14 downto 0) of reg32;

end package detectorfpga_types;
