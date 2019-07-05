----------------------------------------------------------------------------
-- SPI IF
-- Interface to an SPI write only interface of variable length
--
-- expaned to also read data!
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-- 
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity spi_if_write_bits is
	generic( 
		bits:	integer := 24;				-- set to shift register length, should work up to 128
		interfaces: integer := 2;		-- set to number of chips connected
		device_is_DAC : boolean := true -- data is latched on rising edge = true, falling edge = false
	);	
	port(
		clk:					in std_logic;
		reset_n:				in std_logic;
		datain:				in std_logic_vector(bits-1 downto 0);
		wren:					in std_logic_vector(interfaces-1 downto 0);
		busy_n:				out std_logic := '0';	-- default value
		spi_sdi:				out std_logic;
		spi_sclk:			out std_logic;
		spi_load_n:			out std_logic_vector(interfaces-1 downto 0);
		
		spi_sdo:				in std_logic;
		dataout:				out std_logic_vector(bits-1 downto 0)
		);		
end entity spi_if_write_bits;


architecture rtl of spi_if_write_bits is	

type state_type is (waiting, sync_n, writing, sync_h);
signal state : state_type;

signal ckdiv:			 std_logic_vector(3 downto 0); -- we divide the clock (50 MHz) by a factor of 16*2 = 1.5625 MHz
signal cyclecounter:	 std_logic_vector(7 downto 0); -- let's assume we never send more than 128 bits
signal datain_reg	: std_logic_vector(bits-1 downto 0);

constant END_CONDITION : integer := 2*bits-1;

signal wren_last : std_logic_vector(interfaces-1 downto 0);

signal dataout_reg : std_logic_vector(bits downto 0);

begin


process(clk, reset_n)

begin
if(reset_n = '0') then

	ckdiv 			<= (others => '0');
	cyclecounter	<= (others => '0');
	state				<= waiting;	
	datain_reg		<= (others => '0');
	spi_sdi			<= '0';
	spi_sclk			<= '0';
	spi_load_n		<= (others => '0');
	busy_n			<= '0';					-- power up in busy state
	
	wren_last		<= (others => '0');
	
	dataout_reg 	<= (others => '0');
	dataout			<= (others => '0');
	
elsif(clk'event and clk = '1') then
				
	case state is
	
		when waiting =>	
			busy_n		<= '1';				-- indicate that no communication is in progress
			ckdiv 		<= (others => '0');
			cyclecounter<= (others => '0');
			spi_sdi		<= '0';
			spi_sclk		<= '0';				-- Jens: changed this from '1' to '0'
			spi_load_n	<= (others => '1');
			state			<= waiting;
			wren_last	<= wren;				-- we do not write again directly after data has been written

			if(not device_is_DAC)then			
				dataout			<= dataout_reg(bits-1 downto 0);	-- ADC: was correct
			else
				dataout			<= dataout_reg(bits downto 1);	-- DAC: observed left shift by 1, so we move it down again by 1 bit		
			end if;
			
			if(wren = 0 or wren_last = wren)then
			else
				datain_reg	<= datain;		-- data is copied into shiftregister
				state 		<= sync_n;		
				busy_n		<= '0';					
				spi_load_n	<= not wren;	-- set the sync condition, meaning load is driven low
			end if;
			
		when sync_n =>
			spi_sdi	<= datain_reg(bits-1);	-- put the MSB on the line		
			ckdiv 	<= ckdiv + '1';			-- clock division
			if(ckdiv = x"F") then	
				ckdiv 	<= (others => '0');		
				state		<= writing;
			end if;
	
		when writing =>
			ckdiv 	<= ckdiv + '1';
			spi_sdi	<= datain_reg(bits-1);		-- put the MSB on the line	
			dataout_reg(0)	<= spi_sdo;				-- we have the MSB on the line already before clocking for the DACs, not for the ADC
			if(ckdiv = 0) then						-- this is also where this process starts
				cyclecounter 	<= cyclecounter + '1';
				if(cyclecounter(0) = '0') then 	-- create clock -> rising edge
					spi_sclk		<= '1';				-- this is also where this process starts
					dataout_reg	<= dataout_reg(bits-1 downto 0) & '0';		-- at rising edge, new data will be valid, so we shift our current data
					if(not device_is_DAC and cyclecounter /= 0)then
						datain_reg 	<= datain_reg(bits-2 downto 0) & '0';	-- shift at rising edge if data is latched at falling edge (for ADC)
																							-- but only after the first rising edge!
					end if;					
				end if;
				if(cyclecounter(0) = '1') then 	-- create clock -> falling edge
					spi_sclk 	<= '0';
					if(device_is_DAC)then
						datain_reg 	<= datain_reg(bits-2 downto 0) & '0';	-- shift at falling edge if data is latched at rising edge (for DAC)
					end if;
				end if;		
				if(conv_integer(cyclecounter) = END_CONDITION) then -- after having clocked all bits, release the sync condition again
					state 		<= sync_h;									 -- we leave this state with a falling edge!
				end if;					
			end if;	
			
			
		when sync_h =>
			ckdiv 	<= ckdiv + '1';			-- clock division
			if(ckdiv = x"F") then	
				ckdiv 	<= (others => '0');		
				state		<= waiting;		
				spi_load_n	<= (others => '1');	-- raise load signal again, we are done
			end if;
		
		when others =>
			state 	<= waiting;
			
	end case;
end if;
end process;

end rtl;