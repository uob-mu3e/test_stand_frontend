library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.pcie_components.all;
use work.mudaq_registers.all;

entity a10_block is
port (
    io_flash_data       : inout std_logic_vector(31 downto 0);
    o_flash_address     : out   std_logic_vector(31 downto 0);
    o_flash_read_n      : out   std_logic;
    o_flash_write_n     : out   std_logic;
    o_flash_cs_n        : out   std_logic;

    -- i2c

    -- spi
    
    o_nios_hz           : out   std_logic;

    -- pcie ref 100 MHz clocks
    i_pcie0_clk_100     : in    std_logic;
    i_pcie1_clk_100     : in    std_logic;

    -- global 125 MHz clock
    i_clk_125           : in    std_logic;
    i_reset_125_n       : in    std_logic--;
);
end entity;

architecture arch of xcvr_block is

begin

    -- nios

    -- i2c mux
    
    -- xcvr_block 6250 Mbps @ 156.25 MHz

    -- xcvr_block 10000 Mbps @ 250 MHz

    -- pcie0

    -- pcie1

end architecture;
