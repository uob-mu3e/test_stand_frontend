----------------------------------------------------------------------------
-- Mupix Slowcontrol
-- M. Mueller
-- Feb 2022

-- aka "mu3e slowcontrol protocol" .. in need of a better name
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mupix.all;
use work.mudaq.all;
use work.mupix_registers.all;


entity mupix_slowcontrol is
    generic(
        N_CHIPS_PER_SPI_g: positive := 4--;
    );
    port(
        i_clk                   : in  std_logic;
        i_reset_n               : in  std_logic;

        -- connections to config storage
        o_read                  : out mp_conf_array_in(N_CHIPS_PER_SPI_g-1 downto 0);
        i_data                  : in  mp_conf_array_out(N_CHIPS_PER_SPI_g-1 downto 0);
        i_enable                : in  std_logic;

        -- connections to direct spi entity
        o_SIN                   : out std_logic_vector(N_CHIPS_PER_SPI_g-1 downto 0)--;
    );
end entity mupix_slowcontrol;

architecture RTL of mupix_slowcontrol is

begin

    process (i_clk, i_reset_n) is
    begin
        if(i_reset_n = '0') then
          for I in 0 to N_CHIPS_PER_SPI_g-1 loop
                o_read(I).spi_read  <= (others => '0');
                o_read(I).mu3e_read <= (others => '0');
          end loop;
          o_SIN <= (others => '0');
        elsif(rising_edge(i_clk)) then

        end if;
    end process;
end RTL;