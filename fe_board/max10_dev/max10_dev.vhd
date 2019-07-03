library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity max10_dev is
port (

	CLOCK : in std_logic;
	RESET_N : in std_logic;
	
	Arduino_IO13 : in std_logic;
	Arduino_IO12 : out std_logic;
	Arduino_IO11 : out std_logic;
	Arduino_IO10 : out std_logic;
	
	LED1 : out std_logic--;
	
);
end entity;

architecture arch of max10_dev is

	signal adc_clk : std_logic; -- 10 MHz
	
	signal pll_locked : std_logic;

	signal counter : std_logic_vector(31 downto 0) := (others => '0');
	signal nios_pio : std_logic_vector(31 downto 0);
	
begin

	LED1 <= counter(20);

	process(CLOCK)
	begin
		if(rising_edge(CLOCK)) then
			counter <= counter + 1;
		end if;
	end process;
	
	e_nios : component work.cmp.nios
	port map (
			clk_clk     	=> CLOCK,
			pio_export  	=> nios_pio,
			reset_reset_n 	=> RESET_N,
			spi_MISO    	=> Arduino_IO13,
			spi_MOSI    	=> Arduino_IO12,
			spi_SCLK    	=> Arduino_IO11,
			spi_SS_n    	=> Arduino_IO10--,
	);
		
	--- ADC ---
	e_ip_adc : component work.cmp.ip_adc
	port map(clk_clk => CLOCK, reset_reset_n => RESET_N);
	
end architecture;	