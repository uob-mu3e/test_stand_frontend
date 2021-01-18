----------------------------------------------------------------------------
-- Slow Control Unit for MuPix9
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity mp10_slowcontrol is
	port(
		clk:					in std_logic;
		reset_n:				in std_logic;
		ckdiv:				in std_logic_vector(15 downto 0);
		mem_data:			in std_logic_vector(31 downto 0);
		wren:					in std_logic;
		ld_in:				in std_logic;
		rb_in:				in std_logic;
		ctrl_dout:			in std_logic;
		ctrl_din:			out std_logic;	-- to the output! to sync_res
		ctrl_clk1:			out std_logic;
		ctrl_clk2:			out std_logic;
		ctrl_ld:				out std_logic;
		ctrl_rb:				out std_logic;
		busy_n:				out std_logic := '0';	-- default value
		dataout:				out std_logic_vector(31 downto 0)
		);		
end mp10_slowcontrol;


architecture rtl of mp10_slowcontrol is	

-- type state_type is (waiting, rb_hi, data_in, clock_1_hi, clock_1_lo, clock_2_hi, clock_2_lo);
type state_type is (waiting, reset, init, data_in, send_data, sync_reset);
signal state : state_type;

signal ckdiv_fast_reg	: std_logic_vector(3 downto 0);
signal ckdiv_slow_reg	: std_logic_vector(9 downto 0);	-- relativ to fast
signal delay_init_reg	: std_logic_vector(4 downto 0);	-- delay of the init pulse relative to the slow clock
signal duration_init_reg: std_logic_vector(4 downto 0);	-- duration init pulse -- invert
signal cyclecounter:	 std_logic_vector(6 downto 0); 
signal mem_data_reg: std_logic_vector(63 downto 0);
signal wren_last : std_logic;
signal loaded_first_word : std_logic;
signal dataout_reg : std_logic_vector(31 downto 0);

signal ld_in_last: std_logic;
signal rb_in_last: std_logic;

signal slow_clock: std_logic;
signal slow_clock_last: std_logic;

signal ctrl_clk1_reg : std_logic;
signal ctrl_clk1_last : std_logic;

signal ckdiv_fast:std_logic_vector(3 downto 0);
signal ckdiv_slow:std_logic_vector(9 downto 0);
signal init_delay:std_logic_vector(4 downto 0);
signal init_duration:std_logic_vector(4 downto 0);

begin

ctrl_clk1	<= ctrl_clk1_reg;

ckdiv_fast		<=	ckdiv(3 downto 0);
ckdiv_slow		<=	"00" & ckdiv(7 downto 6) & "000" & ckdiv(5 downto 4) &"1";		-- MuPix9!!!   &"100000";
init_delay		<=	ckdiv(11 downto 8)	&"1";				-- partial of "100000" max "011111" min "000001"
init_duration	<=	ckdiv(15 downto 12)	&"1";				-- partial of "100000" max "011111" min "000001", init delay+init_duration should be < "100000"

process(clk, reset_n)

begin
if(reset_n = '0') then
	ctrl_ld		<= '0';
	ctrl_rb		<= '0';
	ctrl_din		<= '0';
	ctrl_clk1_reg	<= '0';
	ctrl_clk1_last	<= '0';
	ctrl_clk2	<= '0';
	busy_n		<= '0';
	dataout		<= (others => '0');
	dataout_reg	<= (others => '0');
	wren_last	<= '0';
	loaded_first_word	<= '0';
	ld_in_last	<= '0';
	rb_in_last	<= '0';
	ckdiv_fast_reg	<= (others => '0');
	ckdiv_slow_reg	<= (others => '0');
	delay_init_reg	<= (others => '0');
	duration_init_reg	<= (others => '0');
	state 		<= waiting;
	slow_clock	<= '0';
	slow_clock_last	<= '0';
	
	mem_data_reg	<= (others => '0');
	cyclecounter	<= (others => '0');
	
elsif(clk'event and clk = '1') then

-- clock division slowcontrol "fast clock" !!constantly running
	ckdiv_fast_reg 	<= ckdiv_fast_reg + '1';	-- clock division
	if(ckdiv_fast_reg >= ckdiv_fast) then				--before concatinated with x"F"
		ckdiv_fast_reg <= (others => '0');		
		ctrl_clk1_reg	<= not ctrl_clk1_reg;
	end if;

	ctrl_clk1_last	<= ctrl_clk1_reg;

	if(ctrl_clk1_reg='1' and ctrl_clk1_last='0') then
		ckdiv_slow_reg 	<= ckdiv_slow_reg + '1';	-- clock division
		if(ckdiv_slow_reg >= ckdiv_slow ) then	
			ckdiv_slow_reg <= (others => '0');	
			slow_clock 		<= not slow_clock;			
		end if;
--		if (ckdiv_slow_reg = "0") then
--			slow_clock 		<= not slow_clock;
--		end if;
	end if;	
	slow_clock_last<= slow_clock;	
	
	
	dataout_reg	<= (others => '0');
	dataout		<= dataout_reg;
				
	case state is
	
		when waiting =>	
			busy_n		<= '1';				-- indicate that no communication is in progress
			cyclecounter<= (others => '0');
			wren_last	<= wren;				-- we do not write again directly after data has been written
			delay_init_reg	<= (others => '0');
			duration_init_reg	<= (others => '0');
			ctrl_din		<= '0';
			ctrl_clk2	<= '0';
			ctrl_ld		<= '0';
			ctrl_rb		<= '0';
			ld_in_last	<= ld_in;
			rb_in_last	<= rb_in;
--			if(ld_in = '1' and ld_in_last = '0')then
--				cyclecounter<= (others => '1');
--				ctrl_ld		<= '1';
--				state			<= clock_2_lo;
--				busy_n		<= '0';	
--			elsif(rb_in = '1' and rb_in_last = '0')then
--				cyclecounter<= (others => '1');
--				ctrl_rb		<= '1';
--				state			<= rb_hi;
--				busy_n		<= '0';	
			if(wren = '1' and wren_last = '0' and loaded_first_word = '0')then
				mem_data_reg(63 downto 32)   <= mem_data;	-- data is copied into shiftregister
				loaded_first_word <= '1';				
			end if;
			if(wren = '1' and wren_last = '0' and loaded_first_word = '1')then
				mem_data_reg(31 downto 0) <= mem_data;	-- data is copied into shiftregister
				state 		<= data_in;		
				busy_n		<= '0';
				loaded_first_word <= '0';				
			end if;
				
			
		when data_in =>
			if(cyclecounter = 0 and mem_data_reg = X"F000000000000000" ) then
				state <= reset;
			elsif(cyclecounter = 0 and mem_data_reg = X"FF00000000000000") then
				state <= sync_reset;
			elsif(slow_clock_last /= slow_clock) then
				state <= init;
			end if;	

		when init =>
			--if(cyclecounter = 0 and ctrl_clk1_last /= ctrl_clk1_reg) then	--ctrl_clk1_last = '0' and ctrl_clk1_reg = '1')
				--delay_init_reg 	<= delay_init_reg + '1';
				--if(delay_init_reg >= init_delay or ckdiv_slow_reg = ckdiv_slow(7 downto 1)) then	
					--delay_init_reg 	<= (others => '0');
					--cyclecounter <= cyclecounter + 1;
					--ctrl_din <= '1';
				--end if;
			if(ctrl_clk1_last = '0' and ctrl_clk1_reg = '1' and ckdiv_slow_reg = ckdiv_slow(7 downto 1) and cyclecounter = 0) then
				ctrl_din <= '1';
				cyclecounter <= cyclecounter + 1;
			elsif(cyclecounter = "000001" and ctrl_clk1_last = '0' and ctrl_clk1_reg = '1' and ckdiv_slow_reg = ckdiv_slow(7 downto 1)) then
				cyclecounter 	<= (others => '0');
				--ctrl_din <= '0';
				state <= send_data;
			end if;
			--elsif(cyclecounter = "000001" and ctrl_clk1_last /= ctrl_clk1_reg) then	--ctrl_clk1_last = '0' and ctrl_clk1_reg = '1')
				--duration_init_reg 	<= duration_init_reg + 1;
				--if(duration_init_reg >= init_duration) then	
					--duration_init_reg 	<= (others => '0');
					--cyclecounter 	<= (others => '0');
					--ctrl_din <= '0';
					--state <= send_data;
				--end if;	
				
				
			--end if;	
			
		when reset =>
			ctrl_clk2				<= '1';
			duration_init_reg 	<= duration_init_reg + '1';	-- clock division
			if(duration_init_reg >= ckdiv(15 downto 12)) then
				duration_init_reg <= (others => '0');
				ctrl_clk2			<= '0';
				state			 		<= waiting;
			end if;

		when sync_reset =>
			ctrl_din	<= '1';
			if(slow_clock_last = '0' and slow_clock = '1') then
				cyclecounter	<= cyclecounter + '1';
				if(cyclecounter >= "1111111") then			--hardcoded 128 slow cycles
					cyclecounter 	<= (others => '0');
					ctrl_din	<= '0';
					state		<= waiting;
				end if;
			end if;

		when send_data =>
			ctrl_din	<= mem_data_reg(0);		--ctrl_din	<= mem_data_reg(31);
			if(ctrl_clk1_last = '0' and ctrl_clk1_reg = '1' and ckdiv_slow_reg = ckdiv_slow(7 downto 1) and cyclecounter < "111111") then  --slow_clock = '0' and ckdiv_slow_reg = ckdiv(11 downto 8)&"010000" --ctrl_clk1_last = '0' and ctrl_clk1_reg = '1')
				cyclecounter	<= cyclecounter + '1';
				--ctrl_din	<= mem_data_reg(31);
				mem_data_reg <= '0' & mem_data_reg(63 downto 1);			--mem_data_reg <= mem_data_reg(30 downto 0) & '0';
			elsif(ctrl_clk1_last = '0' and ctrl_clk1_reg = '1' and ckdiv_slow_reg = ckdiv_slow(7 downto 1)) then	--ctrl_clk1_last = '0' and ctrl_clk1_reg = '1')
				cyclecounter 	<= (others => '0');
				ctrl_din <= '0';
				state <= waiting;
			end if;
		
		when others =>
			state 	<= waiting;
			
	end case;
end if;
end process;

end rtl;
