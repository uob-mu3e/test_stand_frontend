-- Convert gray code to a binary code
-- Sebastian Dittmeier 
-- September 2017
-- dittmeier@physi.uni-heidelberg.de
-- based on code by Niklaus Berger
--
-- takes one clock cycle to do the full thing


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.datapath_components.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity hit_ts_conversion is 
	generic(TS_SIZE 			: integer :=  TIMESTAMPSIZE_MPX8);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		invert_TS			: in std_logic;
		invert_TS2			: in std_logic;
		gray_TS				: in std_logic;
		gray_TS2				: in std_logic;
		hit_in				: in std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_in			: in std_logic;
		hit_out				: out std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_out			: out std_logic
		);
end hit_ts_conversion;

architecture rtl of hit_ts_conversion is

signal hit_in_TS : std_logic_vector(TS_SIZE-1 downto 0);
signal hit_out_TS : std_logic_vector(TS_SIZE-1 downto 0);
signal hit_in_TS2 : std_logic_vector(CHARGESIZE-1 downto 0);
signal hit_out_TS2 : std_logic_vector(CHARGESIZE-1 downto 0);
signal hit_in_reg : std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);

begin

gen_zeros:
if TS_SIZE < TIMESTAMPSIZE_MPX8 generate
	hit_out(TIMESTAMPSIZE_MPX8-1 downto TS_SIZE) <= (others => '0');
end generate gen_zeros;

hit_out(UNPACKER_HITSIZE-1 downto (TIMESTAMPSIZE_MPX8+CHARGESIZE)) <= hit_in_reg(UNPACKER_HITSIZE-1 downto (TIMESTAMPSIZE_MPX8+CHARGESIZE));


with invert_TS select hit_in_TS <=
	hit_in((TS_SIZE-1) downto 0)	 	when '0',
	not hit_in((TS_SIZE-1) downto 0)	when '1';
	
with invert_TS2 select hit_in_TS2 <=
	hit_in((TIMESTAMPSIZE_MPX8+CHARGESIZE-1) downto TIMESTAMPSIZE_MPX8) 		when '0',
	not hit_in((TIMESTAMPSIZE_MPX8+CHARGESIZE-1) downto TIMESTAMPSIZE_MPX8)	when '1';	
	
with gray_TS select hit_out((TS_SIZE-1) downto 0) <=
	hit_in_reg((TS_SIZE-1) downto 0) when '0',
	hit_out_TS when '1';
	
with gray_TS2 select hit_out((TIMESTAMPSIZE_MPX8+CHARGESIZE-1) downto TIMESTAMPSIZE_MPX8) <=
	hit_in_reg((TIMESTAMPSIZE_MPX8+CHARGESIZE-1) downto TIMESTAMPSIZE_MPX8) when '0',
	hit_out_TS2 when '1';	

	i_degray_TS : gray_to_binary 
	generic map(NBITS => TS_SIZE)
	port map(
		reset_n				=> reset_n,
		clk					=> clk,
		gray_in				=> hit_in_TS,
		bin_out				=> hit_out_TS
		);
		
	i_degray_TS2 : gray_to_binary 
	generic map(NBITS => CHARGESIZE)
	port map(
		reset_n				=> reset_n,
		clk					=> clk,
		gray_in				=> hit_in_TS2,
		bin_out				=> hit_out_TS2
		);	
	
process(reset_n, clk)
begin
	if(reset_n = '0')then
		hit_ena_out	<= '0';
	elsif(rising_edge(clk))then
		hit_ena_out <= hit_ena_in;
		hit_in_reg(UNPACKER_HITSIZE-1 downto (TIMESTAMPSIZE_MPX8+CHARGESIZE)) 	<= hit_in(UNPACKER_HITSIZE-1 downto (TIMESTAMPSIZE_MPX8+CHARGESIZE));
		hit_in_reg((TIMESTAMPSIZE_MPX8+CHARGESIZE-1) downto TIMESTAMPSIZE_MPX8) <= hit_in_TS2;
		hit_in_reg((TS_SIZE-1) downto 0) 										<= hit_in_TS;		
	end if;
end process;	

end rtl;
