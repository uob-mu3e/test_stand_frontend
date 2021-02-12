-- Constants solely for use in mupix block

library ieee;
use ieee.std_logic_1164.all;

package mupix_constants is

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

-----------------------------------------------------------
-- mupix ctrl constants
-----------------------------------------------------------

type mp_config_regs_length_t    is array (5 downto 0) of integer;
constant MP_CONFIG_REGS_LENGTH  : mp_config_regs_length_t := (508, 889, 889, 80, 90, 208); -- TODO: check order, assuming bias conf vdac col test tdac here

end package mupix_constants;
