library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;
use work.feb_sc_registers.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

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
        i_flash_fifo_data   : reg32_t;
        i_flash_fifo_we     : reg32_t;
        i_flash_fifo_wclk   : std_logic;
        o_flash_status      : reg32_t   
        );
end max10_controller;      
    
    
architecture RTL of max10_controller is

    signal read_fifo : std_logic;
    signal fifo_empty : std_logic;
    signal fifo_usedw : std_logic_vector(8 downto 0);
    signal fifo_out   : std_logic_vector(7 downto 0);

    type state_type is (idle,adc_reading);
    signal state : state_type;

    signal adccounter : integer;

begin     

    process(i_clk50, i_reset_n)
    begin
    if(i_reset_n = '0') then
        adccounter  <= 0;
        state <= idle;
    elsif(i_clk50'event and i_clk50 = '1') then

        adccounter <= adccounter + 1;

        case state is
        when idle =>
            if(adccounter = adc_read_div)then
                state <= adc_reading;
                adccounter <= 0;
            end if;
        when adc_reading =>
        end case;


    end if;
    end process;    


    fifo_component : altera_mf.altera_mf_components.dcfifo_mixed_widths
    GENERIC MAP (
            add_ram_output_register => "ON",
            intended_device_family => "Arria V",
            lpm_numwords => 64,
            lpm_showahead => "ON",
            lpm_type => "dcfifo",
            lpm_width => 32,
            lpm_widthr => 8,
            lpm_widthu => 7,
            lpm_widthu_r => 9,
            overflow_checking => "ON",
            underflow_checking => "ON",
            use_eab => "ON"
    )
    PORT MAP (
            aclr => not i_reset_n,
            wrclk => i_flash_fifo_wclk,
            data => i_flash_fifo_data, 
            rdreq => read_fifo,
            sclr => not reset_n,
            wrreq => i_flash_fifo_we,
            empty => fifo_empty,
            rdusedw  => fifo_usedw,
            q => fifo_out
    );



end architecture RTL;
