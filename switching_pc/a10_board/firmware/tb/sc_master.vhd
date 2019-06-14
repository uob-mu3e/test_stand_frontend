----------------------------------------------------------------------------
-- Slow Control Master Unit for Switching Board
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity sc_master is
	generic(
		NLINKS : integer :=4
	);
	port(
		clk:				in std_logic;
		reset_n:			in std_logic;
		enable:				in std_logic;
		mem_data_in:		in std_logic_vector(31 downto 0);
		mem_addr:			out std_logic_vector(15 downto 0);
		mem_data_out:		out std_logic_vector(NLINKS * 32 - 1 downto 0);
		mem_data_out_k:		out std_logic_vector(NLINKS * 4 - 1 downto 0);
		done:				out std_logic;
		stateout:			out std_logic_vector(27 downto 0)
		);		
end entity sc_master;

architecture RTL of sc_master is
	
	signal addr_reg	: std_logic_vector(15 downto 0) := (others => '0');
	signal wren_reg	: std_logic_vector(NLINKS-1 downto 0);
	signal cycles		: std_logic_vector(15 downto 0);
	
	type state_type is (waiting, start_wait, starting);
	signal state : state_type;
	
	constant CODE_START : std_logic_vector(11 downto 0) 	:= x"BAD";
	constant CODE_STOP : std_logic_vector(31 downto 0) 	:= x"AFFEAFFE";
	
	signal wait_cnt : std_logic;
	
	signal cycles_cnt : std_logic_vector(15 downto 0);
	
	signal mem_datak : std_logic_vector(3 downto 0); 
	
begin

	mem_addr	 <= addr_reg;
	--mem_addr(15 downto 3) <= (others => '0');
	
	gen_output:
	for I in 0 to NLINKS-1 generate
		process(clk, reset_n)
		begin
			if(reset_n = '0')then
				mem_data_out((I+1)*32-1 downto I*32) <= (others => '0');
				mem_data_out_k((I+1)*4-1 downto I*4) <= (others => '0');
			elsif(rising_edge(clk))then
				mem_data_out((I+1)*32-1 downto I*32) <= x"000000BC";
				mem_data_out_k((I+1)*4-1 downto I*4) <= "0001";
				if (wren_reg(I) = '1') then
					mem_data_out((I+1)*32-1 downto I*32) <= mem_data_in;
					mem_data_out_k((I+1)*4-1 downto I*4) <= mem_datak;
				end if;
			end if;
		end process;
	end generate gen_output;

	process(clk, reset_n)
	begin
		if(reset_n = '0')then
			addr_reg		<= (others => '0');
			wren_reg		<= (others => '0');
			cycles			<= (others => '0');
			cycles_cnt		<= (others => '0');
			state			<= waiting;
			done			<= '0';
			stateout		<= (others => '0');
			wait_cnt		<= '0';
			mem_datak 		<= (others => '0');
		elsif(rising_edge(clk))then
			wait_cnt		<= not wait_cnt;
			mem_datak 		<= (others => '0');
			case state is
			
				when waiting =>
					stateout(3 downto 0) 	<= x"1";
					done					<= '1';			
					wren_reg				<= (others => '0');
					cycles_cnt				<= (others => '0');
				-- toggled register to start
					if(wait_cnt = '0')then
						if(enable = '1')then
							if(mem_data_in(31 downto 20) = CODE_START)then
								state		<= start_wait;
								addr_reg	<= addr_reg + '1';
								done		<= '0';
								cycles		<= mem_data_in(15 downto 0);
							end if;
						end if;
					end if;
				
				when start_wait =>
					stateout(3 downto 0) <= x"2";
					if(wait_cnt = '0')then	
						state		<= starting;
						addr_reg	<= addr_reg + '1';
						mem_datak <= "1000";
						if(mem_data_in(4) = '1') then
							wren_reg <= (others => '1');
						else
							wren_reg	<= (others => '0');
							wren_reg(conv_integer(mem_data_in(3 downto 2)))	<= '1';
						end if;
					end if;
				
				when starting =>
					stateout(3 downto 0) <= x"3";
					if (mem_data_in = CODE_STOP) then
						state		<= waiting;
						--addr_reg <= (others => '0');
						wren_reg	<= (others => '0');
					else
						addr_reg	<= addr_reg + '1';
						cycles_cnt <= cycles_cnt + '1';
					end if;
										
				when others =>
					stateout(3 downto 0) <= x"F";
					state	<= waiting;
					addr_reg <= (others => '0');
					
			end case;
			
		end if;
	end process;

end RTL;
