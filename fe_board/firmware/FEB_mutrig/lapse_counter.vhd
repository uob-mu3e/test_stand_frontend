
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

entity lapse_counter is
generic (
    N_CC : positive := 32--;
);
port (
    i_reset_n   : in  std_logic;
    i_clk       : in  std_logic; -- 125 MHz

    i_en        : in  std_logic;
    i_upper_bnd : in  std_logic_vector(N_CC - 1 downto 0) := (others => '0');
    i_lower_bnd : in  std_logic_vector(N_CC - 1 downto 0);

    i_CC		: in  std_logic_vector(N_CC - 1 downto 0);
    o_CC		: out std_logic_vector(N_CC downto 0)--;
);
end lapse_counter;

architecture arch of lapse_counter is

    signal CC_fpga : std_logic_vector(N_CC - 1 downto 0);
    signal upper, lower : std_logic_vector(N_CC - 1 downto 0);
    signal s_o_CC, s_o_CC_reg : std_logic_vector(N_CC downto 0);
    signal nLapses : integer := 0;

begin

--counting lapsing of coarse counter
--CC lapses every 2^15-1 cycles @ 625MHz. 
p_gen_lapsing: process(i_clk, i_reset_n)
begin
    if ( i_reset_n = '0' ) then
        nLapses <= 0;
        CC_fpga <= (others => '0');
    elsif ( rising_edge(i_clk) ) then
        if ( CC_fpga = 32765 ) then
            nLapses <= nLapses + 1;
            CC_fpga(1 downto 0) <= "11";
            CC_fpga(N_CC - 1 downto 2) <= (others => '0');
        elsif ( CC_fpga = 32763 ) then
            nLapses <= nLapses + 1;
            CC_fpga(0) <= '1';
            CC_fpga(N_CC - 1 downto 1) <= (others => '0');
        elsif ( CC_fpga = 32766 ) then
            nLapses <= nLapses + 1;
            CC_fpga(2 downto 0) <= "100";
            CC_fpga(N_CC - 1 downto 3) <= (others => '0');
        elsif ( CC_fpga = 32764 ) then
            nLapses <= nLapses + 1;
            CC_fpga(1 downto 0) <= "10";
            CC_fpga(N_CC - 1 downto 2) <= (others => '0');
        elsif ( CC_fpga = 32767 ) then
            nLapses <= nLapses + 1;
            CC_fpga(2 downto 0) <= "101";
            CC_fpga(N_CC - 1 downto 3) <= (others => '0');
        else
            CC_fpga <= CC_fpga + 5;
        end if;
    end if;
end process;

s_o_CC  <= "0" & i_CC;

-- assuming that the delay is in the range of 
-- upper=30000 to 32766/7 and 0 to lower=2767
upper <=    std_logic_vector(to_unsigned(30000, upper'length)) when and_reduce(i_upper_bnd) = '0' else
            i_upper_bnd;

lower <=    std_logic_vector(to_unsigned(2767, lower'length)) when and_reduce(i_lower_bnd) = '0' else
            i_lower_bnd;

s_o_CC_reg  <=  s_o_CC - (nLapses - 1) when (i_CC <= 32766 and i_CC >= upper) and (CC_fpga >= 0 and CC_fpga <= lower) else
                s_o_CC - (nLapses + 1) when (i_CC <= lower and i_CC >= 0) and (CC_fpga >= upper and CC_fpga <= 32767) else
                s_o_CC - nLapses;

o_CC        <=  "0" & i_CC when i_en = '0' else
                s_o_CC_reg when s_o_CC_reg <= 32767 - 1 else 
                s_o_CC_reg - 32767 - 1;

--FPGA 2^15-1 counter: 0 1 2 0 1 2 0 1 2
--nLapse:              0 0 0 1 1 1 2 2 2
--FPGA CC_ASIC:        - - 1 2 0 1 2 0 1
--Corrected:           - - 1 2 3 0 1 2 3
--Corrected <= CC - nLapse when CC < FPGA counter else CC - (nLapse - 1);


end architecture;