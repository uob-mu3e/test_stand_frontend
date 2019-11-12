-----------------------------------
--
-- Triple data unpacker 
-- for MuPix8 shared/non-shared links
-- Ann-Kathrin Perrevoort, April 2017
-- 
-- perrevoort@physi.uni-heidelberg.de
--
-- remove timerened dependence
-- Sebastian Dittmeier, September 2017
-- dittmeier@physi.uni-heidelberg.de
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;
use work.datapath_components.all;


entity data_unpacker_triple_new is
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		UNPACKER_HITSIZE	: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
--		reset_out_n			: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		is_shared			: IN STD_LOGIC;
		is_atlaspix			: IN STD_LOGIC;
--		timerend				: IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		hit_out				: OUT STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);		-- Link[7:0] & Row[7:0] & Col[7:0] & Charge[5:0] & TS[9:0]
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);	-- Gray Counter[7:0] & Binary Counter [23:0]
		coarsecounter_ena	: OUT STD_LOGIC;
		link_flag			: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0)
--		error_out			: OUT STD_LOGIC_VECTOR(31 downto 0);
--		readycounter		: OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
		);
end data_unpacker_triple_new;

architecture RTL of data_unpacker_triple_new is

type state_type is (WAITING, SEARCHING, RUNNING);
signal NS 					: state_type	:= WAITING;

signal ready_i				: std_logic_vector(2 downto 0);

type hit_array is array (2 downto 0) of std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
type cnt_array is array (2 downto 0) of std_logic_vector(COARSECOUNTERSIZE-1 downto 0);
type err_array is array (2 downto 0) of std_logic_vector(31 downto 0);
signal hit_out_i				: hit_array;
signal hit_ena_i				: STD_LOGIC_VECTOR(2 downto 0);
signal coarsecounter_i		: cnt_array;
signal coarsecounter_ena_i	: STD_LOGIC_VECTOR(2 downto 0);
signal link_flag_i			: STD_LOGIC_VECTOR(2 downto 0);
signal errorcounter_i		: err_array;

signal datain_reg				: std_logic_vector(7 downto 0);
signal kin_reg					: std_logic;

signal cnt4		: std_logic_vector(1 downto 0);
--signal readycounter_i		: std_logic_vector(31 downto 0)	:= (others => '0');

begin

--readycounter	<= readycounter_i;

genunpack:
FOR i in 2 downto 0 GENERATE
unpacker:data_unpacker_new
	generic map(
		COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
		UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
	)
	port map(
		reset_n				=> reset_n,
--		reset_out_n			=> resets_n(RESET_BIT_UNPACKER_OUT),
		clk					=> clk,
		datain				=> datain_reg,
		kin					=> kin_reg,
		readyin				=> ready_i(i),
		is_atlaspix			=> is_atlaspix,
--		mux_ready			=> mux_ready_i(i),
		hit_out				=> hit_out_i(i),
		hit_ena				=> hit_ena_i(i),
		coarsecounter		=> coarsecounter_i(i),
		coarsecounter_ena	=> coarsecounter_ena_i(i),
		link_flag			=> link_flag_i(i),
		errorcounter		=> errorcounter_i(i)
--		error_out			=> error_out_i((i+1)*32-1 downto i*32)	
		);
		
END GENERATE;

	dataout_proc : process(clk, reset_n)
	
	begin
	
	if(reset_n = '0')then
		hit_out				<= (others => '0');
		hit_ena				<= '0';
		coarsecounter		<= (others => '0');
		coarsecounter_ena	<= '0';
		link_flag			<= '0';
	elsif(rising_edge(clk))then
		hit_ena				<= hit_ena_i(2) or hit_ena_i(1) or hit_ena_i(0);
		coarsecounter_ena	<= coarsecounter_ena_i(2) or coarsecounter_ena_i(1) or coarsecounter_ena_i(0);		
		link_flag			<= link_flag_i(2) or link_flag_i(1) or link_flag_i(0);
		-- these signals will never be high at the same time
		if(coarsecounter_ena_i(0) = '1')then
			coarsecounter	<= coarsecounter_i(0);
		elsif(coarsecounter_ena_i(1) = '1')then
			coarsecounter	<= coarsecounter_i(1);
		elsif(coarsecounter_ena_i(2) = '1')then
			coarsecounter	<= coarsecounter_i(2);
		end if;
		if(hit_ena_i(0) = '1')then
			hit_out	<= hit_out_i(0);
		elsif(hit_ena_i(1) = '1')then
			hit_out	<= hit_out_i(1);
		elsif(hit_ena_i(2) = '1')then
			hit_out	<= hit_out_i(2);
		end if;		
	end if;
	
	end process;
	
--
--timerend_nonshared	<= to_integer(unsigned(timerend));
--
--timerend_shared		<=	0 when ((timerend_nonshared=0) or (timerend_nonshared=1)) else	-- prevent negative values
--								timerend_nonshared-2;
	
	fsmProc: process (clk, reset_n)
	begin
	
		if reset_n = '0' then
		
			NS 			<= WAITING;
			cnt4			<= "00";
			ready_i		<= "000";
			datain_reg	<= (others => '0');
			kin_reg		<= '0';
			errorcounter<= (others => '0');
			
		elsif rising_edge(clk) then

			errorcounter <= errorcounter_i(0) + errorcounter_i(1) + errorcounter_i(2);
		
			datain_reg	<= datain;
			kin_reg		<= kin;
				
			if(readyin = '0')then

				NS 		<= WAITING;
				cnt4		<= "00";
				ready_i	<= "000";				
			else

				case NS is
					when WAITING =>
						cnt4		<= "00";
						ready_i	<= "000";
						if( (kin = '1') and (datain = k28_5))then
							NS <= SEARCHING;
						end if;
						
					when SEARCHING => 						
						if ( (kin = '0') or (kin = '1' and datain = k28_0) ) then
							ready_i	<= "001";
							cnt4		<= "00";
							NS			<= RUNNING;
						end if;
						
					when RUNNING =>
						cnt4 <= cnt4 + 1;
						if(cnt4 = "11")then
							if(is_shared = '1')then
								ready_i	<= ready_i(1 downto 0) & ready_i(2);
							end if;
						end if;
											
					when others =>
						NS <= WAITING;
	
				end case;	-- NS
			end if;	-- readyin

		end if;
		
	end process;


end RTL;
