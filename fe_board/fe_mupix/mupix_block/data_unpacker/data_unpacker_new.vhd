-----------------------------------
--
-- Data unpacker for MuPix8
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
use work.daq_constants.all;




entity data_unpacker_new is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		UNPACKER_HITSIZE	: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		is_atlaspix			: IN STD_LOGIC;
		hit_out				: OUT STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);		-- Link[7:0] & Row[7:0] & Col[7:0] & Charge[5:0] & TS[9:0]
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);	-- Gray Counter[7:0] & Binary Counter [23:0]
		coarsecounter_ena	: OUT STD_LOGIC;
		link_flag			: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0)
		);
end data_unpacker_new;

architecture RTL of data_unpacker_new is

type state_type is (IDLE, ERROR, COUNTER, LINK, DATA);--BINCNT, DATA);
signal NS 					: state_type;

signal data_i				: std_logic_vector(31 downto 0)	:= (others => '0');

signal errorcounter_reg	: std_logic_vector(31 downto 0);
--signal data_error 		: std_logic_vector(15 downto 0);
--signal k_error				: std_logic_vector(1 downto 0);	
--signal datain_reg			: std_logic_vector(7 downto 0);
--signal kin_reg				: std_logic;

signal link_i				: std_logic_vector(7 downto 0);
signal link_reg			: std_logic_vector(7 downto 0);
signal link_toggle		: std_logic					:= '0';
signal data_mode			: std_logic;

signal cnt4					: std_logic_vector(1 downto 0);

signal counter_seen		: std_logic;
signal coarse_reg			: std_logic;
signal hit_reg				: std_logic;
signal link_flag_reg		: std_logic;
--signal bincnt_int			: integer range 0 to 3	:= 0;
--signal data_int			: integer range 0 to 3	:= 0;
--signal counter_int		: integer range 0 to 3	:= 0;
	
begin

errorcounter 	<= errorcounter_reg;
--error_out		<= x"00" & "000" & k_error(1) & data_error(15 downto 8) & "000" & k_error(0) & data_error(7 downto 0);

-- unpacker should do the following things:
-- ignore K28.5 completely
-- to create hits: we first have to find a link identifier
-- after link identifier: we get either a counter + hits or a link identifier
-- as long as we did not find a link identifier, all non-k data will be interpreted as counters (send counter mode)

-- when IDLE 
--		if k=1 und K28.5 => IDLE
--		if k=1 und K28.0 => LINK
--		if k=0 => COUNTER
		
-- when LINK
--		if k=0 			  => COUNTER
--		if k=1 und K28.0 => LINK

-- when COUNTER
-- 	if k=0 und LINK seen	=> HIT
-- 	if k=0 und no LINK 	=> COUNTER
--		if k=1 und K28.0 		=> LINK

-- when HIT
--		if k=0 				=> HIT
--		if k=1 und K28.0 	=> LINK
	
	fsmProc: process (clk, reset_n)
	begin
		if reset_n = '0' then
			NS <= IDLE;
			
			hit_ena	<= '0';
			hit_out	<= (others => '0');
			coarsecounter		<= (others => '0');
			coarsecounter_ena	<= '0';
			link_flag			<= '0';
			
			link_flag_reg	<= '0';
			coarse_reg		<= '0';
			hit_reg			<= '0';

			link_i		<= (others =>'0');
			link_reg		<= (others =>'0');
			link_toggle	<= '0';
			
			errorcounter_reg	<= (others => '0');

			cnt4				<= "00";
			data_mode		<= '0';	-- indicates if all counter mode or actual hit data
			counter_seen	<= '0';
		elsif rising_edge(clk) then

			link_flag_reg	<= '0';
			coarse_reg		<= '0';
			hit_reg			<= '0';
			
			hit_ena				<= hit_reg or coarse_reg;
			coarsecounter_ena	<= coarse_reg;
			link_flag			<= link_flag_reg;
			
			coarsecounter 	<= data_i(7 downto 0) & data_i(31 downto 8);-- gray counter & binary counter
			if(coarse_reg = '1')then
				hit_out 		<= data_i(31 downto 8) & link_i(3 downto 0) & x"3" & data_i(7 downto 0);	-- binary counter & link & x"3" & gray counter	
			elsif(hit_reg = '1')then
				if(is_atlaspix = '1')then
					hit_out		<= link_i & data_i(7 downto 0) & data_i(31 downto 24) & data_i(23 downto 18) & data_i(8) & data_i(9) & data_i(10) & data_i(11) & data_i(12) & data_i(13) & data_i(14) & data_i(15) & data_i(16) & data_i(17);	-- Link & Row & Col & Charge & TS
				else
					hit_out		<= link_i & data_i(7 downto 0) & data_i(15 downto 8) & data_i(21 downto 16) & data_i(31 downto 22);	-- Link & Row & Col & Charge & TS
				end if;
			end if;
			
			if(readyin = '0')then
				link_reg		<= (others => '0');
				link_toggle	<= '0';
				NS				<= IDLE;
			else
			
				data_i	<= data_i(23 downto 0) & datain;

					case NS is
					
						when IDLE =>
							cnt4			<= "00";
							link_toggle	<= '0';
							if kin = '0'then									-- counter mode
								if(data_mode = '1' and counter_seen = '1')then	-- we expect valid hit data here after LINK ID and counter was seen
									NS		<= DATA;
									cnt4	<= "01";
								else
									NS 	<= COUNTER;
									cnt4	<= "01";
								end if;		
							elsif kin = '1' and datain = k28_0 then	-- data mode
								NS <= LINK;
							elsif kin = '1' and datain = k28_5 then
								NS <= IDLE;		
							else
								NS <= ERROR;
							end if;


						when COUNTER =>
							if kin = '0' then
								cnt4	 <= cnt4 + '1';
								if cnt4 = "11" then
									coarse_reg	<= '1';
									counter_seen<= '1';
									NS 			<= IDLE;
--									coarsecounter_ena <= '1';
--									coarsecounter <= datain & data_i(31 downto 8);								-- gray counter & binary counter
--									hit_ena <= '1';																		-- Also send the counter as a hit
--									hit_out <= data_i(31 downto 8) & link_i(3 downto 0) & x"3" & datain;	-- binary counter & link & x"3" & gray counter		
								end if;
							elsif kin = '1' and datain = k28_5 then	--and counter_int = 3 then
								NS	<= IDLE;
							else
								NS <= ERROR;
--								data_error 	<= datain & datain_reg;
--								k_error 		<= kin & kin_reg;
							end if;
			
						when LINK =>
							if kin = '0' then
								if link_toggle = '0' then
									link_reg			<= datain;
								elsif link_reg = datain then
									link_i			<= link_reg;
									link_flag_reg	<= '1';
									link_toggle		<= '0';
									NS					<= IDLE;
									data_mode		<= '1';
									counter_seen	<= '0';
								else
									NS				<= ERROR;
--									data_error 	<= datain & datain_reg;
--									k_error 		<= kin & kin_reg;
								end if;
							elsif kin = '1' and datain = k28_0 then
								link_toggle	<= '1';
							else
								NS <= ERROR;
--								data_error 	<= datain & datain_reg;
--								k_error 		<= kin & kin_reg;
							end if;

							
						when DATA =>
						
							if kin = '0' then
								cnt4	 <= cnt4 + '1';
								if cnt4 = "11" then
									hit_reg		<= '1';
									NS 			<= IDLE;
								end if;
							elsif kin = '1' and datain = k28_5 then
								NS 			<= IDLE;
							else
								NS 			<= ERROR;
							end if;
	
						when ERROR =>
							errorcounter_reg <= errorcounter_reg + '1';
							if ( kin = '1' and datain = k28_5 ) then
								NS <= IDLE;
							else	
								NS <= ERROR;
--								data_error 	<= datain & datain_reg;
--								k_error 		<= kin & kin_reg;					
							end if;
						
						when others =>
							NS <= IDLE;

		
					end case;	-- NS
			end if;	-- readyin
			
		end if;
		
	end process;


end RTL;
