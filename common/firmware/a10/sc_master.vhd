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
end entity;

architecture rtl of sc_master is

    --integer to one hot encoding
    function to_onehot(a:integer ; n:integer) return std_logic_vector is
    variable res: std_logic_vector(n-1 downto 0):=(others =>'0');
    begin
            res(a):='1';
            return res;
    end function;

	
	signal addr_reg	: std_logic_vector(15 downto 0) := (others => '0');
	signal wren_reg	: std_logic_vector(15 downto 0);--)(NLINKS-1 downto 0);
	
	type state_type is (waiting, start_wait, starting, get_start_add, get_length);
	signal state : state_type;
	
	constant CODE_START : std_logic_vector(11 downto 0) 	:= x"BAD";
	constant CODE_STOP : std_logic_vector(31 downto 0) 	:= x"0000009C";
	
	signal wait_cnt : std_logic;
	
	signal mem_datak : std_logic_vector(3 downto 0); 

	signal mask_addr : std_logic_vector(15 downto 0); 
	signal length : std_logic_vector(15 downto 0);
	signal cur_length : std_logic_vector(15 downto 0);
	
begin

	mem_addr	 <= addr_reg;
	
	gen_output:
	for I in 0 to NLINKS-1 generate
		process(clk, reset_n)
		begin
			if(reset_n = '0')then
				mem_data_out((I+1)*32-1 downto I*32) <= (others => '0');
				mem_data_out_k((I+1)*4-1 downto I*4) <= (others => '0');
			elsif(rising_edge(clk))then
				if (wren_reg(I) = '1') then
					mem_data_out((I+1)*32-1 downto I*32) <= mem_data_in;
					mem_data_out_k((I+1)*4-1 downto I*4) <= mem_datak;
				else 
                    mem_data_out((I+1)*32-1 downto I*32) <= x"000000BC";
                    mem_data_out_k((I+1)*4-1 downto I*4) <= "0001";
				end if;
			end if;
		end process;
	end generate gen_output;

	process(clk, reset_n)
	begin
		if(reset_n = '0')then
			addr_reg		<= (others => '0');
			wren_reg		<= (others => '0');
			state			<= waiting;
			done			<= '0';
			stateout		<= (others => '0');
			wait_cnt		<= '0';
			mem_datak 		<= (others => '0');
			length 			<= (others => '0');
			cur_length		<= (others => '0');
		elsif(rising_edge(clk))then
			wait_cnt		<= not wait_cnt;
			mem_datak 	<= (others => '0');
			wren_reg		<= (others => '0');
			stateout(3 downto 0)		<= x"F";
			
			if (addr_reg = x"FFFF") then
				addr_reg		<= (others => '0');
			end if;
			
			case state is
			
				when waiting =>
					stateout(3 downto 0) <= x"1";
					done						<= '1';
					if(wait_cnt = '0')then
						if(enable = '1')then
							if(mem_data_in(31 downto 20) = CODE_START)then
								state		<= start_wait;
								addr_reg	<= addr_reg + '1';
								done		<= '0';
							end if;
						end if;
					end if;
				
				when start_wait =>
					stateout(3 downto 0) <= x"2";
					if(wait_cnt = '0')then	
						state		<= get_start_add;
						addr_reg	<= addr_reg + '1';
						mem_datak <= "0001";
						mask_addr <= mem_data_in(23 downto 8); -- get fpga id if x"FFFF" write to all links, if 1 first link and so on
						if(mem_data_in(23 downto 8) = x"FFFF") then
							wren_reg <= (others => '1');
						else
                            wren_reg <= mem_data_in(23 downto 8); -- todo fix me to more the 16 addr one hot
						end if;
					end if;
					
				when get_start_add =>
					stateout(3 downto 0) <= x"4";
					if(wait_cnt = '0')then
						if(mask_addr(15 downto 0) = x"FFFF") then
							wren_reg <= (others => '1');
						else
							wren_reg <= mask_addr(15 downto 0);
						end if;
						addr_reg	<= addr_reg + '1';
						state		<= get_length;
					end if;
					
				when get_length =>
					stateout(3 downto 0) <= x"4";
					if(wait_cnt = '0')then
						if(mask_addr(15 downto 0) = x"FFFF") then
							wren_reg <= (others => '1');
						else
							wren_reg <= mask_addr(15 downto 0);
						end if;
						addr_reg	<= addr_reg + '1';
						-- length is only for data words so trailer is not in
						length 		<= mem_data_in(15 downto 0);
						cur_length	<= x"0001";
						state		<= starting;
					end if;
				
				when starting =>
					stateout(3 downto 0) <= x"5";
					if(wait_cnt = '0')then
						if(mask_addr(15 downto 0) = x"FFFF") then
							wren_reg <= (others => '1');
						else
							wren_reg <= mask_addr(15 downto 0);
						end if;
						if (length + '1' = cur_length) then
							mem_datak <= "0001";
							state		<= waiting;
							addr_reg	<= addr_reg + '1';
							length	<= (others => '0');
							cur_length	<= x"0001";
						else
							cur_length <= cur_length + '1';
							addr_reg	<= addr_reg + '1';
						end if;
					end if;
										
				when others =>
					state		<= waiting;
					addr_reg <= (others => '0');
					wren_reg	<= (others => '0');
					cur_length	<= (others => '0');
					length	<= (others => '0');
					cur_length	<= x"0001";
			end case;
			
		end if;
	end process;

end architecture;
