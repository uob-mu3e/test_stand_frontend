library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity max10_dev is
port (

	CLOCK : in std_logic; -- 50 MHz
	reset : in std_logic;
	
	Arduino_IO13 : in std_logic;
	Arduino_IO12 : out std_logic;
	Arduino_IO11 : out std_logic;
	Arduino_IO10 : out std_logic;
	
	SWITCH1 : in std_logic;
	SWITCH2 : in std_logic;
	SWITCH3 : in std_logic;
	SWITCH4 : in std_logic;
	
	LED1 : out std_logic;
	LED2 : out std_logic;
	LED3 : out std_logic;
	LED4 : out std_logic;
	LED5 : out std_logic--;
	
);
end entity;

architecture arch of max10_dev is

	signal adc_clk : std_logic; -- 10 MHz
	signal nios_clk : std_logic; -- 50 MHz
	
	signal pll_locked : std_logic;
	
	signal reset_n : std_logic;
	
	signal sw : std_logic_vector(2 downto 0);
	signal led : std_logic_vector(2 downto 0);

	signal counter_0 : std_logic_vector(31 downto 0) := (others => '0');
	signal counter_1 : std_logic_vector(31 downto 0) := (others => '0');
	signal nios_pio : std_logic_vector(31 downto 0);
	
begin

	sw(0) 	<= SWITCH1;
	sw(1) 	<= SWITCH2;
	sw(2) 	<= SWITCH3;
	LED3		<= led(0);
	LED4		<= led(1);
	LED5		<= SWITCH4;
	reset_n 	<= not reset;

	--- PLL ---
	e_ip_altpll : component work.cmp.ip_altpll
	port map (
		areset 	=> reset,
		inclk0 	=> CLOCK,
		c0 		=> adc_clk,
		c1			=> nios_clk,
		locked 	=> pll_locked--,
	);

	--- LEDs ---
	LED1 <= not counter_0(20);
	LED2 <= not counter_1(20);

	process(CLOCK)
	begin
		if(rising_edge(CLOCK)) then
			counter_0 <= counter_0 + 1;
		end if;
	end process;
	
	process(adc_clk)
	begin
		if(rising_edge(adc_clk)) then
			counter_1 <= counter_1 + 1;
		end if;
	end process;
	
	--- NIOS ---
	e_nios : component work.cmp.nios
	port map (
			adc_clock_clk     	=> adc_clk,
			adc_locked_export  	=> pll_locked,
			clk_clk           	=> CLOCK,
			led_io_export     	=> led,
			pio_export        	=> nios_pio,
			reset_reset_n     	=> reset_n,
			spi_MISO          	=> Arduino_IO13,
			spi_MOSI          	=> Arduino_IO12,
			spi_SCLK          	=> Arduino_IO11,
			spi_SS_n          	=> Arduino_IO10,
			sw_io_export      	=> sw--,
	);
	
end architecture;	