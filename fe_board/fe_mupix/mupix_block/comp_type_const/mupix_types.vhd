-- Types for use in mupix telescope

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;
use work.daq_constants.all;

package mupix_types is

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
type reg_array                  is array (NCHIPS-1 downto 0) of reg32;

end package mupix_types;
