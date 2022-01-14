-- FEB reg mapping for the firefly transceivers
-- M.Mueller, Jan 2021


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;


entity firefly_reg_mapping is
port (
    i_clk156                    : in  std_logic;

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0)--;
);
end entity;

architecture rtl of firefly_reg_mapping is


begin

    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then 
            
        elsif(rising_edge(i_clk156)) then

            regaddr             := to_integer(unsigned(i_reg_add));
            o_reg_rdata         <= x"CCCCCCCC";

        end if;
    end process;

end architecture;
