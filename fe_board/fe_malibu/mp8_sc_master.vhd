----------------------------------------------------------------------------
-- Slow Control Master Unit for MuPix8
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity mp8_sc_master is
	generic(NCHIPS : integer :=4);
	port(
		clk:					in std_logic;
		reset_n:				in std_logic;
		mem_data_in:		in std_logic_vector(31 downto 0);
		busy_n:				in std_logic_vector(NCHIPS-1 downto 0);
--		start:				in std_logic;
		
		mem_addr:			out std_logic_vector(15 downto 0);
		mem_data_out:		out std_logic_vector(31 downto 0);
		wren:					out std_logic_vector(NCHIPS-1 downto 0);
		ctrl_ld:				out std_logic_vector(NCHIPS-1 downto 0);
		ctrl_rb:				out std_logic_vector(NCHIPS-1 downto 0);
		done:					out std_logic;
		stateout:			out std_logic_vector(27 downto 0)
		);		
end entity mp8_sc_master;

architecture RTL of mp8_sc_master is
	
	signal addr_reg	: std_logic_vector(15 downto 0);
	signal wren_reg	: std_logic_vector(NCHIPS-1 downto 0);
	signal ld_reg		: std_logic_vector(NCHIPS-1 downto 0);
	signal rb_reg		: std_logic_vector(NCHIPS-1 downto 0);
	signal cycles		: std_logic_vector(14 downto 0);
	
	type state_type is (waiting, start_wait, starting, write_cmd, write_wait, writing, continue_wait, continue);
	signal state : state_type;
	
	constant CODE_START : std_logic_vector(31 downto 0) 	:= x"BADC0DED";
	constant CODE_STOP : std_logic_vector(31 downto 0) 	:= x"ABAD1DEA";
	
--	signal start_reg : std_logic;
	
	signal start_addr : std_logic_vector(15 downto 0);
	
	signal wait_cnt : std_logic_vector(3 downto 0);
	signal stateout_reg:		 std_logic_vector(27 downto 0);
	
begin

	mem_addr	<= addr_reg;
	stateout	<= stateout_reg;

	process(clk, reset_n)
	begin
		if(reset_n = '0')then
		
			addr_reg	<= (others => '0');
			wren_reg	<= (others => '0');
			cycles	<= (others => '0');
			state		<= waiting;
--			start_reg<= '0';
			done		<= '0';
			mem_data_out	<= (others => '0');
			wren		<= (others => '0');
			ctrl_ld	<= (others => '0');
			ctrl_rb	<= (others => '0');
			start_addr	<= (others => '0');
			ld_reg	<= (others => '0');
			rb_reg	<= (others => '0');
			
			stateout_reg	<= (others => '0');
			
			wait_cnt	<= (others => '0');
		elsif(rising_edge(clk))then
		
--			start_reg 	<= start;
			
			wait_cnt		<= wait_cnt + 1;
			stateout_reg<= stateout_reg;
			
			case state is
			
				when waiting =>
					stateout_reg(3 downto 0) <= x"1";
					done		<= '1';			
					wren		<= (others => '0');
					ctrl_ld	<= (others => '0');
					ctrl_rb	<= (others => '0');
				-- toggled register to start
				--	if((start_reg = '0') and (start = '1') and ((not busy_n) = 0))then
					if(wait_cnt = x"0")then
						if((not busy_n) = 0)then
							if(mem_data_in = CODE_START)then
								state		<= start_wait;
								addr_reg	<= addr_reg + 1;
								done		<= '0';
								start_addr	<= addr_reg;
							end if;
						end if;
					end if;
				
				when start_wait =>
					stateout_reg(3 downto 0) <= x"2";
					if(wait_cnt = x"0" )then	
						if(mem_data_in(31) = '0')then -- MSB is not allowed to be 1!
							state		<= starting;
							addr_reg	<= addr_reg + 1;
							
							wren_reg	<= mem_data_in(NCHIPS-1 downto 0);
							ld_reg	<= mem_data_in(2*NCHIPS-1 downto NCHIPS);
							rb_reg	<= mem_data_in(3*NCHIPS-1 downto 2*NCHIPS);
							cycles	<= mem_data_in(30 downto 16);
						else
							state <= waiting;
						end if;
					end if;
				
				when starting =>
					stateout_reg	<= (others => '0');
					stateout_reg(3 downto 0)   <= x"3";
					stateout_reg(NCHIPS-1+4 downto 4)   <= wren_reg(NCHIPS-1 downto 0);
					stateout_reg(NCHIPS-1+8 downto 8)   <= ld_reg(NCHIPS-1 downto 0);
					stateout_reg(26 downto 12) <= cycles;
					if(wait_cnt = x"0" )then	
						state		<= write_cmd;
					end if;
					
				when write_cmd =>
					stateout_reg(3 downto 0) <= x"4";
					ctrl_rb 	<= rb_reg;
					if(rb_reg = 0)then
						if((not busy_n) = 0)then
							wren 				<= wren_reg and busy_n;
							mem_data_out	<= mem_data_in;					
							addr_reg			<= addr_reg + 1;
							if(cycles /= 0)then
								cycles		<= cycles - 1;
							end if;
							state 			<= write_wait;
						end if;
					else
						if((rb_reg and busy_n) = 0)then
							rb_reg	<= (others => '0');
						end if;
					end if;
					
				when write_wait =>
					stateout_reg(3 downto 0) <= x"5";
					if((wren_reg and busy_n) = 0)then	-- all channels that should write (wren_reg) are now busy!
						state				<= writing;
					end if;
					
				when writing =>
					stateout_reg(3 downto 0) <= x"6";
					wren	<= (others => '0');
					if((not busy_n) = 0)then
						if(cycles /= 0)then
							state	<= write_cmd;
						elsif(mem_data_in = CODE_STOP)then
							done		<= '1';
							ctrl_ld	<= ld_reg;
							state		<= waiting;--continue_wait;
							addr_reg <= addr_reg + 1;
						end if;
					end if;
					
				when continue_wait =>
					stateout_reg(3 downto 0) <= x"7";
					ctrl_ld	<= (others => '0');
					state 	<= waiting;--continue;	-- maybe continuing is a problem?
					
				when continue => -- to prevent that we write more than one full memory block!
					stateout_reg(3 downto 0) <= x"8";
					if(not busy_n = 0)then
						if(addr_reg /= start_addr and mem_data_in = CODE_START and (not busy_n = 0))then							
							state		<= start_wait;
							addr_reg	<= addr_reg + 1;
							done		<= '0';
						else
	--						ctrl_ld	<= wren_reg;
							state		<= waiting;					
						end if;
					end if;
					
						
				when others =>
					stateout_reg(3 downto 0) <= x"F";
					state	<= waiting;
					
					
			end case;
			
		end if;
	end process;

end RTL;
