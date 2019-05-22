----------------------------------------------------------------------------
-- Slow Control Slave Unit for Switching Board
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity sc_slave is
	generic(
		NLINKS : integer :=4
	);
	port(
		clk:				in std_logic;
		reset_n:			in std_logic;
		enable:				in std_logic;
		link_data_in:		in std_logic_vector(31 downto 0);
		link_data_in_k:		in std_logic_vector(3 downto 0);
		mem_data_out:		out std_logic_vector(31 downto 0);
		mem_addr_out:		out std_logic_vector(15 downto 0);
		mem_wren:			out std_logic;			
		done:				out std_logic;
		stateout:			out std_logic_vector(27 downto 0)
		);		
end entity sc_slave;

architecture RTL of sc_slave is

	signal mem_data_o : std_logic_vector(31 downto 0);
	signal mem_addr_o : std_logic_vector(15 downto 0);
	signal mem_wren_o: std_logic;

	constant CODE_START : std_logic_vector(11 downto 0) := x"BCC";
	constant CODE_STOP : std_logic_vector(31 downto 0) 	:= x"AFFEAFFE";

	type state_type is (waiting, starting);
	signal state : state_type;
	
begin

	mem_data_out <= mem_data_o;
	mem_addr_out <= mem_addr_o;
	mem_wren	 <= mem_wren_o;

	memory : process(reset_n, clk)
	begin
	if(reset_n = '0')then
		mem_data_o <= (others => '0');
		mem_addr_o <= (others => '0');
		stateout <= (others => '0');
		mem_wren_o <= '0';
		
	elsif(rising_edge(clk))then

		mem_wren_o <= '0';

		if(link_data_in = x"000000BC" and link_data_in_k(0) = '1') then
			stateout(3 downto 0) <= x"F";
			mem_wren_o <= '0';
		else
		
		case state is
			
				when waiting =>

					if(link_data_in(31 downto 20) = CODE_START and link_data_in_k(3) = '1') then
				 		stateout(3 downto 0) <= x"1";
				 		mem_data_o <= link_data_in;
				 		mem_wren_o <= '1';
				 		state <= starting;
				 	end if;

				when starting =>
					if (link_data_in = CODE_STOP) then
						stateout(3 downto 0) <= x"2";
				 		mem_data_o <= (others => '0');
				 		mem_wren_o <= '0';
				 		state <= waiting;
				 	else
				 		mem_addr_o <= mem_addr_o + '1';
						stateout(3 downto 0) <= x"3";
						mem_data_o <= link_data_in;
						mem_wren_o <= '1';
					end if;

				when others =>
					stateout(3 downto 0) <= x"E";
					mem_data_o <= (others => '0');
					mem_wren_o <= '0';
		end case;

		end if;
	end if;
	end process;

end RTL;
