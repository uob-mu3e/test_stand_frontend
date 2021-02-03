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

constant TIMESTAMPSIZE_MP10     :  integer := 11;
constant TIMESTAMPSIZE          :  integer := 11;

subtype TSRANGE                 is integer range TIMESTAMPSIZE-1 downto 0;

constant MHITSIZE               :  integer := HITSIZE+2;

constant COARSECOUNTERSIZE      :  integer := 32;

subtype  COLRANGE               is integer range 23 downto 16;
subtype  ROWRANGE               is integer range 15 downto 8;

constant CHIPRANGE              :  integer := 3;

-----------------------------------------------------------
-----------------------------------------------------------

constant BINCOUNTERSIZE         :  integer := 24;
constant UNPACKER_HITSIZE       :  integer := 32;
constant CHARGESIZE_MP10        :  integer := 5;

constant NOTSHITSIZE            :  integer := HITSIZE -TIMESTAMPSIZE;
subtype NOTSRANGE               is integer range HITSIZE-1 downto TIMESTAMPSIZE;

constant HITSORTERBINBITS       :  integer := 4;
constant H                      :  integer := HITSORTERBINBITS;
constant HITSORTERADDRSIZE      :  integer := TIMESTAMPSIZE + HITSORTERBINBITS;

constant BITSPERTSBLOCK         :  integer := 4;
subtype TSBLOCKRANGE            is integer range TIMESTAMPSIZE-1 downto BITSPERTSBLOCK;
subtype TSNONBLOCKRANGE         is integer range BITSPERTSBLOCK-1 downto 0;

constant COMMANDBITS            :  integer := 20;

constant COUNTERMEMADDRSIZE     :  integer := 8;
constant NMEMS                  :  integer := 2**(TIMESTAMPSIZE-COUNTERMEMADDRSIZE);
constant COUNTERMEMDATASIZE     :  integer := 5;
subtype COUNTERMEMSELRANGE      is integer range TIMESTAMPSIZE-1 downto COUNTERMEMADDRSIZE;
subtype COUNTERMEMADDRRANGE     is integer range COUNTERMEMADDRSIZE-1 downto 0;

-- Bit positions in the counter fifo of the sorter
subtype MEMCOUNTERRANGE        is integer range 2*NCHIPS*HITSORTERBINBITS-1 downto 0;
constant MEMOVERFLOWBIT        :  integer := 2*NCHIPS*HITSORTERBINBITS;
constant HASMEMBIT             :  integer := 2*NCHIPS*HITSORTERBINBITS+1;
subtype TSINFIFORANGE           is integer range HASMEMBIT+TIMESTAMPSIZE downto HASMEMBIT+1;
subtype TSBLOCKINFIFORANGE      is integer range TSINFIFORANGE'left downto TSINFIFORANGE'left-BITSPERTSBLOCK+1;
subtype TSINBLOCKINFIFORANGE    is integer range TSINFIFORANGE'right+BITSPERTSBLOCK-2  downto TSINFIFORANGE'right;
subtype SORTERFIFORANGE         is integer range TSINFIFORANGE'left downto 0;

constant NSORTERCOUNTERS        : integer := 40;

end package mupix_constants;
