
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


-- This entity should correct the early lapsing of the Mutrig cc counter
-- assuming the Mutrig should count from 0 to 3 but it only counts from 0 to 2 
-- we have the following "correction":

-- FPGA 2^15-1 counter: 0 1 2 0 1 2 0 1 2
-- nLapse:              0 0 0 1 1 1 2 2 2
-- FPGA CC_ASIC:        - - 1 2 0 1 2 0 1
-- Corrected:           - - 1 2 3 0 1 2 3
-- Corrected <= CC - nLapse when CC < FPGA counter else CC - (nLapse - 1);

-- The correction is done by counting according to the Mutrig by increasing a counter
-- on the FPGA by 5 when we count with the 125MHz clk.

entity lapse_counter is
generic (
    N_CC : positive := 32--;
);
port (
    i_reset_n   : in  std_logic; -- i_run_state(RUN_STATE_BITPOS_SYNC)
    i_clk       : in  std_logic; -- 125 MHz

    i_en        : in  std_logic; -- this entity can be enabled disabled via software
    i_upper_bnd : in  std_logic_vector(N_CC - 1 downto 0); -- upper bnd for the correction (default: 30000)

    i_CC        : in  std_logic_vector(N_CC - 1 downto 0); -- counter from the Mutrig @625MHz
    o_CC        : out std_logic_vector(N_CC - 1 downto 0)  -- corrected Mutrig counter 
);
end lapse_counter;

architecture arch of lapse_counter is

    --signal CC_fpga : std_logic_vector(N_CC - 1 downto 0);
    --signal  : unsigned(N_CC - 1 downto 0);
    --signal  : std_logic_vector(N_CC downto 0);
    --signal nLapses : natural : natural range 0 to 32767 := 0;
    --signal s_i_CC : std_logic_vector(N_CC downto 0);
    signal s_o_CC, s_o_CC_reg : unsigned(N_CC - 1 downto 0);
    signal upper, lower, CC_fpga, nLapses : natural range 0 to 32767 := 0;

begin

-- counting lapsing of coarse counter
-- CC lapses every 2^15-1 cycles @ 625MHz. 
p_gen_lapsing: process(i_clk, i_reset_n)
begin
    if ( i_reset_n = '0' ) then
        nLapses <= 0;
        CC_fpga <= 0;
    elsif ( rising_edge(i_clk) ) then
        -- NOTE: In the following we have 5 different edge cases where
        -- the internal CC_fpga counter would count wrongly
        if ( CC_fpga = 32765 ) then
            nLapses <= nLapses + 1;
            CC_fpga <= 3;
        elsif ( CC_fpga = 32763 ) then
            nLapses <= nLapses + 1;
            CC_fpga <= 1;
        elsif ( CC_fpga = 32766 ) then
            nLapses <= nLapses + 1;
            CC_fpga <= 4;
        elsif ( CC_fpga = 32764 ) then
            nLapses <= nLapses + 1;
            CC_fpga <= 2;
        elsif ( CC_fpga = 32767 ) then
            nLapses <= nLapses + 1;
            CC_fpga <= 5;
        else
            CC_fpga <= CC_fpga + 5;
        end if;
        if (s_o_CC  <= 32766 and s_o_CC  >= upper and CC_fpga <= lower) then
            report "s_o_CC " & work.util.to_hstring(std_logic_vector(s_o_CC));
            report "nLapses " & work.util.to_hstring(std_logic_vector(to_unsigned(nLapses, o_CC'length)));
            report "s_o_CC_reg " & work.util.to_hstring(std_logic_vector(s_o_CC_reg));
        end if;
    end if;
end process;


-- assuming that the default delay is in the range of 
-- upper=30000 to 32766/7 and 0 to lower=2767
-- get upper lower bnd
upper       <=  to_integer(unsigned(i_upper_bnd));
lower       <=  32767 - to_integer(unsigned(i_upper_bnd));

-- TODO: I put ranges here bcz its easier to read
-- TODO: tb add check if nLapses is higher than s_o_CC, if nLapse is exp: 2^20 (cc range)
-- TODO: this should never happen, debug counter
s_o_CC      <=  unsigned(i_CC);

s_o_CC_reg  <=  s_o_CC - (nLapses - 1) when s_o_CC  <= 32766 and s_o_CC  >= upper and CC_fpga <= lower and nLapses > 0 else
                s_o_CC - (nLapses + 1) when CC_fpga <= 32767 and CC_fpga >= upper and s_o_CC  <= lower else
                s_o_CC - nLapses;

o_CC        <=  i_CC when i_en = '0' else
                std_logic_vector(s_o_CC_reg);

end architecture;
