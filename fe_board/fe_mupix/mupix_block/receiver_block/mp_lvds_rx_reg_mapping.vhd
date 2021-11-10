-- mupix lvds rx reg mapping
-- M. Mueller, Nov 2021

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;
entity mp_lvds_rx_reg_mapping is
port (
    i_clk156                    : in  std_logic;
    i_reset_n                   : in  std_logic;

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    i_lvds_status               : in  work.util.slv32_array_t    (35 downto 0) := (others => x"CCCCCCCC")--;

);
end entity;

architecture rtl of mp_lvds_rx_reg_mapping is
        signal lvds_status              : work.util.slv32_array_t(35 downto 0);
    begin
    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then 

        elsif(rising_edge(i_clk156)) then
            regaddr                     := to_integer(unsigned(i_reg_add));
            lvds_status                 <= i_lvds_status;
            -----------------------------------------------------------------
            ---- lvds rx regs -----------------------------------------------
            -----------------------------------------------------------------
            for I in 0 to MUPIX_LVDS_STATUS_BLOCK_LENGTH-1 loop 
                if ( regaddr = I + MP_LVDS_STATUS_START_REGISTER_W and i_reg_re = '1' ) then
                    o_reg_rdata <= lvds_status(MP_LINK_ORDER(I));
                end if;
            end loop;

        end if;
    end process;
end architecture;
