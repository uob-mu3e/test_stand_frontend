library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;
use work.feb_sc_registers.all;

entity max10_controller is
    generic(
        adc_read_div : integer := 5000000; -- 10 Hz
    )
    port(
        -- Max10 SPI
        o_SPI_cs    : out std_logic;
        o_SPI_clk   : out std_logic;
        io_SPI_mosi : inout std_logic;
        io_SPI_miso : inout std_logic;
        io_SPI_D1   : inout std_logic;
        io_SPI_D2   : inout std_logic;
        io_SPI_D3   : inout std_logic;
    
        -- Interface to Arria
        i_clk50      : in std_logic;
        i_reset_n    : in std_logic;

        -- Slow control stuff (MAX10 ADCs)
        o_adc       : reg32array_t( 4 downto 0);

        -- Programming the SPI flash on the MAX10
        i_flash_start_addr  : reg32_t;
        i_flash_data        : reg32_t;
        i_flash_we          : reg32_t;
        o_flash_status      : reg32_t   
        );
end max10_controller;      
    
    
architecture RTL of max10_controller is


begin     





end architecture RTL;
