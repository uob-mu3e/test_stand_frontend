-- Types for use in mupix telescope

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;

package mupix_types is

subtype reg32 		is std_logic_vector(31 downto 0);
type reg32array	is array (NREGISTERS-1 downto 0) of reg32;

subtype reg64 		is std_logic_vector(63 downto 0);
subtype REG64_TOP_RANGE is integer range 63 downto 32;
subtype REG64_BOTTOM_RANGE is integer range 31 downto 0;
subtype reg62 		is std_logic_vector(61 downto 0);
subtype REG62_TOP_RANGE is integer range 61 downto 31;
subtype REG62_BOTTOM_RANGE is integer range 30 downto 0;

type reg64array 	is array (NCHIPS-1 downto 0) of reg64;
type reg48array 	is array (NCHIPS-1 downto 0) of std_logic_vector(47 downto 0);

-- reduced to help with timing hopefully
type trigger_scaler_array	is array (NTRIGGERS downto 0) of std_logic_vector(15 downto 0);

type injection_counter_array	is array (NINJECTIONS*NCHIPS-1 downto 0) of reg32;

type hitsorter_debug_array	is array (7 downto 0) of reg32;

subtype readmemaddrtype 		is std_logic_vector(15 downto 0);

subtype chipmarkertype 			is std_logic_vector(7 downto 0);

subtype byte_t is std_logic_vector(7 downto 0);
type inbyte_array is array (NLVDS-1 downto 0) of byte_t;

type state_type is (INIT, START, PRECOUNT, COUNT);

type chips_reg32	is array (NCHIPS-1 downto 0) of reg32;
type gxlinks_reg32 is array(NGX-1 downto 0) of reg32;
type links_reg32 is array(NCHIPS*4-1 downto 0) of reg32;
type links_bytearray is array(15 downto 0) of std_logic_vector(7 downto 0);

type chips_histo_reg32 is array (0 to 7) of chips_reg32;
type histo_col_array is array(0 to 7) of std_logic_vector(7 downto 0);

constant HISTO_ROW_START: histo_col_array := (x"00", x"04", x"08", x"0C", x"10", x"00", x"07", x"13");
constant HISTO_ROW_END: histo_col_array 	:= (x"03", x"07", x"0B", x"0F", x"13", x"00", x"07", x"13");
constant HISTO_COL_START: histo_col_array := (x"00", x"08", x"10", x"18", x"20", x"28", x"30", x"38");
constant HISTO_COL_END: histo_col_array 	:= (x"07", x"0F", x"17", x"1F", x"27", x"2F", x"37", x"3F");

type chips_vec33	is array (NCHIPS-1 downto 0) of std_logic_vector(32 downto 0);
type chips_vec8	is array (NCHIPS-1 downto 0) of std_logic_vector(7 downto 0);
type links_vec33	is array (NCHIPS*4-1 downto 0) of std_logic_vector(32 downto 0);
type links_vec8	is array (NCHIPS*4-1 downto 0) of std_logic_vector(7 downto 0);
type links_vec10	is array (NCHIPS*4-1 downto 0) of std_logic_vector(9 downto 0);
type links_vec9	is array (NCHIPS*4-1 downto 0) of std_logic_vector(8 downto 0);

subtype vecdata is std_logic_vector(COARSECOUNTERSIZE+HITSIZE+2-1 downto 0);
type chips_vecdata is array (NCHIPS-1 downto 0) of vecdata;

type fifo_init_array is array (NLVDS-1 downto 0) of std_logic_vector(1 downto 0);
type fifo_usedw_array is array(NLVDS-1 downto 0) of std_logic_vector(3 downto 0);

-- for pseudo data generator
type NumCOL_array is array (2 downto 0) of integer range 0 to 200;
type MatrixSEL_array is array (2 downto 0) of std_logic_vector(1 downto 0);

end package mupix_types;