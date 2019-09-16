library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity top is
port (

	CLOCK : in std_logic; -- 50 MHz
	RESET_N : in std_logic;

	Arduino_IO13 : in std_logic;
	Arduino_IO12 : out std_logic;
	Arduino_IO11 : out std_logic;
	Arduino_IO10 : out std_logic--;
    
    --Arduino_A0  : in std_logic;
----
--	SWITCH1 : in std_logic;
----	SWITCH2 : in std_logic;
----	SWITCH3 : in std_logic;
----	SWITCH4 : in std_logic;
----    SWITCH5 : in std_logic;
--
--
--	LED1 : out std_logic;
--	LED2 : out std_logic;
--	LED3 : out std_logic;
--	LED4 : out std_logic;
--	LED5 : out std_logic--;

);
end entity;

architecture arch of top is

--    component my_altmult_add IS
--	PORT
--	(
--		clock0		: IN STD_LOGIC  := '1';
--		dataa_0		: IN STD_LOGIC_VECTOR (11 DOWNTO 0) :=  (OTHERS => '0');
--		datab_0		: IN STD_LOGIC_VECTOR (11 DOWNTO 0) :=  (OTHERS => '0');
--		result		: OUT STD_LOGIC_VECTOR (63 DOWNTO 0)
--	);
--    END component my_altmult_add;

    constant count_max : unsigned(24 downto 0) := (others => '1');
    
	signal adc_clk : std_logic; -- 10 MHz
	signal nios_clk : std_logic; -- 50 MHz

	signal pll_locked : std_logic;
    signal adc_response_valid : std_logic;

	signal sw : std_logic_vector(4 downto 0);
	signal led : std_logic_vector(4 downto 0);

	signal nios_pio : std_logic_vector(31 downto 0);
    
    signal adc_data : std_logic_vector(11 downto 0);
    signal adc_data_store : std_logic_vector(11 downto 0);

    signal x2 : STD_LOGIC_VECTOR(63 DOWNTO 0);
    
    signal e_x,e_x2 : unsigned(63 downto 0);
    
    signal counter : unsigned(24 downto 0);
    signal mult_delay_cnt : unsigned(10 downto 0);
    
begin

--    sw(0)   <= SWITCH1;
----	sw(1)   <= SWITCH2;
----	sw(2)   <= SWITCH3;
----    sw(3)   <= SWITCH4;
----    sw(4)   <= SWITCH5;
--    
--	LED1    <= led(0);
--    LED2    <= e_x(0);
--    LED3    <= e_x2(0);
--    LED4    <= counter(0);
--    LED5    <= led(0);

    
--    process(adc_clk,reset_n)
--    begin
--        if (reset_n='0') then
--            e_x <= (others => '0');
--            e_x2 <= (others => '0');
--            counter <= (others => '0');
--            
--        elsif rising_edge(adc_clk) then
--        
--            if mult_delay_cnt = 2 then 
--                e_x2 <= e_x2 + unsigned(x2);
--            end if;  
--            
--            if (counter < count_max) then
--                mult_delay_cnt <= mult_delay_cnt + 1;
--                led(0) <= '1';
--                if adc_response_valid = '1' then
--                    mult_delay_cnt <= (others => '0');
--                    e_x <= e_x + unsigned(adc_data);
--                    counter <= counter + 1;
--                end if;  
--            else
--            
--                led(0) <= '0';
--                
--            end if;
--        end if;
--    end process;
--
--    --- Multiplier ---
--    e_mult : my_altmult_add
--	port map (
--		
--			clock0		=> adc_clk,
--			dataa_0		=> adc_data,
--			datab_0		=> adc_data,
--			result		=> x2
--	);
--
-- 
--    ---ADC---
--    testadc : component work.cmp.myadc
--    port map(
--			adc_pll_clock_clk      => adc_clk,
--			adc_pll_locked_export  => pll_locked,           -- export
--			clock_clk              => adc_clk,             -- clk
--			command_valid          => '1',                -- valid
--			command_channel        => "00001",              -- channel
--			command_startofpacket  => '1',                -- startofpacket
--			command_endofpacket    => '1',                -- endofpacket
--			command_ready          => open,                 -- ready
--			reset_sink_reset_n     => reset_n,              -- reset_n
--			response_valid         => adc_response_valid,   -- valid
--			response_channel       => open,                 -- channel
--			response_data          => adc_data,             -- data
--			response_startofpacket => open,                 -- startofpacket
--			response_endofpacket   => open--,                 -- endofpacket
--		);
--
--    
--
--	--- PLL ---
--	e_ip_altpll : entity work.ip_altpll
--	port map (
--		areset => not reset_n,
--		inclk0 => CLOCK,
--		c0     => adc_clk,
--		c1     => nios_clk,
--		locked => pll_locked--,
--	);
    
       
        --    process(adc_clk, reset_n)
--    begin
--        if (reset_n ='0') then
--            led <= (others => '0');
--            adc_data_store <= (others => '0');
--        elsif rising_edge(adc_clk) then 
--            if adc_response_valid = '1' then
--                adc_data_store <= adc_data;
--            end if; 
--            if (sw(1) = '1') then
--                led <=  adc_data_store(4 downto 0);
--            elsif ( sw(2) = '1') then
--                led <= adc_data_store(9 downto 5);
--            else 
--                led <= "000" & adc_data_store(11 downto 10);
--            end if;
--        end if;
--    end process;
----    
--    channel <= "10001" when SWITCH4='1' else "00001";
--
-- 	   

    	--- NIOS ---
	e_nios : component work.cmp.nios
	port map (
			clk_clk           => CLOCK,
			led_io_export     => led,
			pio_export        => nios_pio,
			rst_reset_n       => reset_n,
			spi_MISO          => Arduino_IO13,
			spi_MOSI          => Arduino_IO12,
			spi_SCLK          => Arduino_IO11,
			spi_SS_n          => Arduino_IO10,
			sw_io_export      => sw--,
	);

end architecture;
