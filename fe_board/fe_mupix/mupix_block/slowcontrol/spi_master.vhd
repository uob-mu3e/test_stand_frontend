----------------------------------------------------------------------------
-- SPI Master Interface
-- Controls the 4xDAC + 1xDAC + 1xADC of the MuPix8 TestBoard
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
-- April 2017
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity spi_master is
	port(
		clk:						in std_logic;
		reset_n:					in std_logic;
		injection1_reg:		in std_logic_vector(15 downto 0);
		threshold_pix_reg:	in std_logic_vector(15 downto 0);
		threshold_low_reg:	in std_logic_vector(15 downto 0);
		threshold_high_reg:	in std_logic_vector(15 downto 0);
		temp_dac_reg:			in std_logic_vector(15 downto 0);
		temp_adc_reg:			in std_logic_vector(15 downto 0);		
		wren:						in std_logic_vector(2 downto 0);
		busy_n:					out std_logic := '0';	-- default value
		spi_sdi:					out std_logic;
		spi_sclk:				out std_logic;
		spi_load_n:				out std_logic_vector(2 downto 0);
		
		spi_sdo:					in std_logic_vector(2 downto 0);
		injection1_out:		out std_logic_vector(15 downto 0);
		threshold_pix_out:	out std_logic_vector(15 downto 0);
		threshold_low_out:	out std_logic_vector(15 downto 0);
		threshold_high_out:	out std_logic_vector(15 downto 0);
		temp_dac_out:			out std_logic_vector(15 downto 0);
		temp_adc_out:			out std_logic_vector(31 downto 0)		
		);		
end entity spi_master;


architecture rtl of spi_master is	

signal datain_dac4X : 	std_logic_vector(63 downto 0);
signal dataout_dac4X : 	std_logic_vector(63 downto 0);
signal wren_master : 	std_logic_vector(2 downto 0);
signal slave_busy_n : 	std_logic_vector(2 downto 0);
signal slave_spi_sdi :  std_logic_vector(2 downto 0);
signal slave_spi_sclk : std_logic_vector(2 downto 0);
signal temp_adc_r	:		std_logic_vector(31 downto 0);

signal cycle : integer range 0 to 2;	-- three interfaces, counts up to two
signal wren_reg : std_logic_vector(2 downto 0);

type state_type is (check, increment, start_writing, stop_writing);
signal state : state_type;


begin

master_proc : process(clk, reset_n)
begin

	if(reset_n = '0')then
		busy_n			<= '0';
		wren_master		<= (others => '0');
		wren_reg			<= (others => '0');
		cycle				<= 0;
		
		state				<= check;
		
	elsif(rising_edge(clk))then
		
		-- as we use the regwritten signal, we catch the case of consecutive or parallel write commands 
		wren_reg	<= wren_reg or wren;
		
		case state is
		
			when check =>
				if(slave_busy_n = "111")then	-- no communication is going on, not busy
					busy_n	<= '1';

					if(wren_reg(cycle)='1')then			-- this device should start communication
						wren_master			<= (others => '0');	-- no device selected						
						wren_master(cycle)<= wren_reg(cycle);	-- selects this specific device	
						state 				<= start_writing;						
					else
						state <= increment;
					end if;
				else 									-- should not occur, apart from reset
					busy_n 	<= '0';				-- indicate that IF is busy and check again
					state		<= check;
				end if;
			
			when increment =>
				if(cycle < 2)then
					cycle	<= cycle + 1;
				else
					cycle	<= 0;
				end if;
				state <= check;
			
			when start_writing =>
				wren_reg(cycle) <= '0';					-- this slave is used, so we "acknowlegde" and clear the write request
				if(slave_busy_n(cycle) = '0')then	-- slave is responsive and starts communication
					wren_master	<= (others => '0');
					busy_n		<= '0';
					state			<= stop_writing;
				end if;
			
			when stop_writing =>
				if(slave_busy_n(cycle) = '1')then	-- slave has stopped communication
					state <= increment;
				end if;
			
			when others => 			
				state <= check;
				
			end case;
	end if;

end process;

spi_sdi	<= slave_spi_sdi(cycle);	-- replace by mux?	
spi_sclk	<= slave_spi_sclk(cycle);	-- replace by mux?	

datain_dac4X			<= threshold_high_reg & threshold_low_reg & threshold_pix_reg & injection1_reg;
injection1_out			<= dataout_dac4X(15 downto 0);
threshold_pix_out		<= dataout_dac4X(31 downto 16);
threshold_low_out		<= dataout_dac4X(47 downto 32);
threshold_high_out	<= dataout_dac4X(63 downto 48);

dac4X : work.spi_if_write_bits
	generic map( 	
		bits => 64,				
		interfaces =>  1,		
		device_is_DAC => true		
	)
	port map(
		clk			=> clk, 
		reset_n		=> reset_n, 
		datain		=> datain_dac4X,
		wren			=> wren_master(0 downto 0),
		busy_n		=> slave_busy_n(0),
		spi_sdi		=> slave_spi_sdi(0),
		spi_sclk		=> slave_spi_sclk(0),
		spi_load_n	=> spi_load_n(0 downto 0),
		spi_sdo		=> spi_sdo(0),
		dataout		=> dataout_dac4X
		);		
		
dacTemp : work.spi_if_write_bits
	generic map( 	
		bits => 16,				
		interfaces =>  1,		
		device_is_DAC => true		
	)
	port map(
		clk			=> clk, 
		reset_n		=> reset_n, 
		datain		=> temp_dac_reg,
		wren			=> wren_master(1 downto 1),
		busy_n		=> slave_busy_n(1),
		spi_sdi		=> slave_spi_sdi(1),
		spi_sclk		=> slave_spi_sclk(1),
		spi_load_n	=> spi_load_n(1 downto 1),
		spi_sdo		=> spi_sdo(1),
		dataout		=> temp_dac_out
		);	

temp_adc_r <= temp_adc_reg & temp_adc_reg;		
		
adcTemp : work.spi_if_write_bits
	generic map( 	
		bits => 32,				
		interfaces =>  1,		
		device_is_DAC => false		
	)
	port map(
		clk			=> clk, 
		reset_n		=> reset_n, 
		datain		=> temp_adc_r,
		wren			=> wren_master(2 downto 2),
		busy_n		=> slave_busy_n(2),
		spi_sdi		=> slave_spi_sdi(2),
		spi_sclk		=> slave_spi_sclk(2),
		spi_load_n	=> spi_load_n(2 downto 2),
		spi_sdo		=> spi_sdo(2),
		dataout		=> temp_adc_out
		);			

end rtl;
