-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Hit multiplexer 4-1
-- Assume hits at maximum every fourth cycle
-- Niklaus Berger, Feb 2014
-- Ann-Kathrin Perrevoort, Oct 2015
-- 
-- nberger@physi.uni-heidelberg.de
-- perrevoort@physi.uni-heidelberg.de

-- based on hit_mulitplexer.vhd
-- deals with hits and coarse counters at the same time
--
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;



entity hit_serializer is 
	generic(
	SERIALIZER_HITSIZE : integer := UNPACKER_HITSIZE+4;
	SERBINCOUNTERSIZE	: integer := BINCOUNTERSIZE+4;
	SERHITSIZE : integer := 40
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		hit_in1				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena1				: IN STD_LOGIC;
		hit_in2				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena2				: IN STD_LOGIC;
		hit_in3				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena3				: IN STD_LOGIC;
		hit_in4				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena4				: IN STD_LOGIC;
		time_in1				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena1			: IN STD_LOGIC;
		time_in2				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena2			: IN STD_LOGIC;
		time_in3				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena3			: IN STD_LOGIC;
		time_in4				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena4			: IN STD_LOGIC;
		hit_out				: OUT STD_LOGIC_VECTOR (SERHITSIZE-1 DOWNTO 0);
		hit_ena				: OUT STD_LOGIC;
		time_out				: OUT STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena				: OUT STD_LOGIC
		);
end hit_serializer;

architecture RTL of hit_serializer is

	signal ena 		 : std_logic_vector(3 downto 0);
	signal ena_del1 : std_logic_vector(3 downto 0);
	signal ena_del2 : std_logic_vector(3 downto 0);
	signal ena_del3 : std_logic_vector(3 downto 0);
	
	signal ena_del1_nors : std_logic_vector(3 downto 0);
	signal ena_del2_nors : std_logic_vector(3 downto 0);
	signal ena_del3_nors : std_logic_vector(3 downto 0);
	
	signal hit1		 : STD_LOGIC_VECTOR (SERHITSIZE-1 DOWNTO 0);
	signal hit2		 : STD_LOGIC_VECTOR (SERHITSIZE-1 DOWNTO 0);
	signal hit3		 : STD_LOGIC_VECTOR (SERHITSIZE-1 DOWNTO 0);
	signal hit4		 : STD_LOGIC_VECTOR (SERHITSIZE-1 DOWNTO 0);
	
	signal timeena 		: std_logic_vector(3 downto 0);
	signal timeena_del1 	: std_logic_vector(3 downto 0);
	signal timeena_del2 	: std_logic_vector(3 downto 0);
	signal timeena_del3 	: std_logic_vector(3 downto 0);
	
	signal time1	 : STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
	signal time2	 : STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
	signal time3	 : STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
	signal time4	 : STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);

begin

process(clk, reset_n)
begin
if(reset_n = '0') then
	hit_ena		<= '0';
	ena			<= (others => '0');
	ena_del1		<= (others => '0');
	ena_del2		<= (others => '0');
	ena_del3		<= (others => '0');
	ena_del1_nors <= (others => '0');
	ena_del2_nors <= (others => '0');
	ena_del3_nors <= (others => '0');
	time_ena			<= '0';
	timeena 			<= (others => '0');
	timeena_del1	<= (others => '0');
	timeena_del2	<= (others => '0');
	timeena_del3	<= (others => '0');
	
	hit1		<= (others => '0');
	hit2		<= (others => '0');
	hit3		<= (others => '0');
	hit4		<= (others => '0');
	hit_out 	<= (others => '0');
elsif(clk'event and clk = '1') then
	
	ena				<= ((hit_ena4 & hit_ena3 & hit_ena2 & hit_ena1) and (not ena)) and (not ena_del1_nors);
	ena_del1			<= ena;
	ena_del2			<= ena_del1;
	ena_del3			<= ena_del2;
	
	ena_del1_nors	<= ena;
	ena_del2_nors	<= ena_del1_nors;
	ena_del3_nors	<= ena_del2_nors;
	
	timeena			<= time_ena4 & time_ena3 & time_ena2 & time_ena1;
	timeena_del1	<= timeena;
	timeena_del2	<= timeena_del1;
	timeena_del3	<= timeena_del2;
	
	if(hit_ena1 = '1' and ena(0) = '0' and ena_del1_nors(0) = '0') then
		hit1	<= hit_in1(SERIALIZER_HITSIZE-1 downto SERIALIZER_HITSIZE-4) & hit_in1(SERIALIZER_HITSIZE-9 downto 0);	-- take half of the LinkID bits, i.e. x"A" instead of x"AA"
		time1	<= time_in1;
	end if;
	
	if(hit_ena2 = '1' and ena(1) = '0' and ena_del1_nors(1) = '0') then
		hit2	<= hit_in2(SERIALIZER_HITSIZE-1 downto SERIALIZER_HITSIZE-4) & hit_in2(SERIALIZER_HITSIZE-9 downto 0);
		time2	<= time_in2;
	end if;
	
	if(hit_ena3 = '1' and ena(2) = '0' and ena_del1_nors(2) = '0') then
		hit3	<= hit_in3(SERIALIZER_HITSIZE-1 downto SERIALIZER_HITSIZE-4) & hit_in3(SERIALIZER_HITSIZE-9 downto 0);
		time3	<= time_in3;
	end if;
	
	if(hit_ena4 = '1' and ena(3) = '0' and ena_del1_nors(3) = '0') then
		hit4	<= hit_in4(SERIALIZER_HITSIZE-1 downto SERIALIZER_HITSIZE-4) & hit_in4(SERIALIZER_HITSIZE-9 downto 0);
		time4	<= time_in4;
	end if;
	
	if(ena_del3(0) = '1') then
		hit_out		<= hit1;
		hit_ena		<= '1';
		time_out		<= time1;
		time_ena		<= timeena_del3(0);
	elsif (ena_del3(1) = '1') then
		hit_out		<= hit2;
		hit_ena		<= '1';
		time_out		<= time2;
		time_ena		<= timeena_del3(1);
	elsif (ena_del3(2) = '1') then
		hit_out		<= hit3;
		hit_ena		<= '1';
		time_out		<= time3;
		time_ena		<= timeena_del3(2);
	elsif (ena_del3(3) = '1') then
		hit_out		<= hit4;
		hit_ena		<= '1';	
		time_out		<= time4;
		time_ena		<= timeena_del3(3);
	elsif(ena_del2(0) = '1') then
		hit_out		<= hit1;
		hit_ena		<= '1';
		ena_del3(0)	<= '0';
		time_out		<= time1;
		time_ena		<= timeena_del2(0);
	elsif (ena_del2(1) = '1') then
		hit_out		<= hit2;
		hit_ena		<= '1';
		ena_del3(1)	<= '0';
		time_out		<= time2;
		time_ena		<= timeena_del2(1);
	elsif (ena_del2(2) = '1') then
		hit_out		<= hit3;
		hit_ena		<= '1';
		ena_del3(2)	<= '0';
		time_out		<= time3;
		time_ena		<= timeena_del2(2);
	elsif (ena_del2(3) = '1') then
		hit_out		<= hit4;
		hit_ena		<= '1';	
		ena_del3(3)	<= '0';
		time_out		<= time4;
		time_ena		<= timeena_del2(3);
	elsif(ena_del1(0) = '1') then
		hit_out		<= hit1;
		hit_ena		<= '1';
		ena_del2(0)	<= '0';
		time_out		<= time1;
		time_ena		<= timeena_del1(0);
	elsif (ena_del1(1) = '1') then
		hit_out		<= hit2;
		hit_ena		<= '1';
		ena_del2(1)	<= '0';
		time_out		<= time2;
		time_ena		<= timeena_del1(1);
	elsif (ena_del1(2) = '1') then
		hit_out		<= hit3;
		hit_ena		<= '1';
		ena_del2(2)	<= '0';
		time_out		<= time3;
		time_ena		<= timeena_del1(2);
	elsif (ena_del1(3) = '1') then
		hit_out		<= hit4;
		hit_ena		<= '1';
		ena_del2(3)	<= '0';	
		time_out		<= time4;
		time_ena		<= timeena_del1(3);
	elsif(ena(0) = '1') then
		hit_out		<= hit1;
		hit_ena		<= '1';
		ena_del1(0)	<= '0';
		time_out		<= time1;
		time_ena		<= timeena(0);
	elsif (ena(1) = '1') then
		hit_out		<= hit2;
		hit_ena		<= '1';
		ena_del1(1)	<= '0';
		time_out		<= time2;
		time_ena		<= timeena(1);
	elsif (ena(2) = '1') then
		hit_out		<= hit3;
		hit_ena		<= '1';
		ena_del1(2)	<= '0';
		time_out		<= time3;
		time_ena		<= timeena(2);
	elsif (ena(3) = '1') then
		hit_out		<= hit4;
		hit_ena		<= '1';
		ena_del1(3)	<= '0';
		time_out		<= time4;
		time_ena		<= timeena(3);
	else
		hit_ena		<= '0';
		time_ena		<= '0';
	end if;

end if;
end process;

end rtl;