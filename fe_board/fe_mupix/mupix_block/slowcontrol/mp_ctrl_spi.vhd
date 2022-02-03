----------------------------------------------------------------------------
-- Mupix SPI 
-- M. Mueller
-- JAN 2022
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mupix.all;
use work.mudaq.all;


entity mp_ctrl_spi is
    generic( 
        DIRECT_SPI_FIFO_SIZE_g: positive := 4;
        N_CHIPS_PER_SPI_g: positive := 4--;
    );
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic--;
    );
end entity mp_ctrl_spi;

architecture RTL of mp_ctrl_spi is


begin

    process (i_clk, i_reset_n) is
    begin
        if(i_reset_n = '0') then

        elsif(rising_edge(i_clk)) then
         
        end if;
    end process;
end RTL;