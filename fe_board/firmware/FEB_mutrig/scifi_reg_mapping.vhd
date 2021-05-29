-- M.Mueller, May 2021

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.scifi_registers.all;

entity scifi_reg_mapping is
generic (
    N_LINKS : positive := 1--;
);
port (
    i_clk_156                   : in  std_logic;
    i_reset_n                   : in  std_logic;

    i_reg_add                   : in  std_logic_vector(7 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    o_cntreg_ctrl               : out std_logic_vector(31 downto 0);

    i_cntreg_num                : in  std_logic_vector(31 downto 0);
    i_cntreg_denom_b            : in  std_logic_vector(63 downto 0);
    i_rx_pll_lock               : in  std_logic;
    i_frame_desync              : in  std_logic_vector(1 downto 0)--;
    --i_rx_dpa_lock_reg           : 

);
end entity;

architecture rtl of scifi_reg_mapping is


begin

    process(i_clk_156)
        variable regaddr : integer;
    begin

    if i_reset_n = '0' then
        
    elsif rising_edge(i_clk_156) then
        o_reg_rdata         <= X"CCCCCCCC";
        regaddr             := to_integer(unsigned(i_reg_add(7 downto 0)));
    end if;
end architecture;
