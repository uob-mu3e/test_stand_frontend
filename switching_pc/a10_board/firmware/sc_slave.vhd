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
generic (
		NLINKS : integer :=4
);
port (
		clk:                in std_logic;
		reset_n:            in std_logic;
		enable:             in std_logic;
		link_data_in:       in std_logic_vector(NLINKS * 32 - 1 downto 0);
		link_data_in_k:     in std_logic_vector(NLINKS * 4 - 1 downto 0);
		mem_data_out:       out std_logic_vector(31 downto 0);
		mem_addr_out:       out std_logic_vector(15 downto 0);
		mem_addr_finished_out:       out std_logic_vector(15 downto 0);
		mem_wren:           out std_logic;
		stateout:           out std_logic_vector(3 downto 0)
);
end entity;

architecture RTL of sc_slave is

	signal mem_data_o : std_logic_vector(31 downto 0);
	signal mem_addr_o : std_logic_vector(15 downto 0);
	signal mem_wren_o : std_logic;
	signal current_link : integer range 0 to NLINKS - 1;
	

	type state_type is (init, waiting, starting);
	signal state : state_type;

begin

	mem_data_out <= mem_data_o;
	mem_addr_out <= mem_addr_o;
	mem_wren     <= mem_wren_o;

	memory : process(reset_n, clk)
	begin
	if(reset_n = '0')then
		mem_data_o <= (others => '0');
		mem_addr_o <= (others => '1');
		mem_addr_finished_out <= (others => '1');
		stateout <= (others => '0');
		mem_wren_o <= '0';
		current_link <= 0;
		state <= waiting;--init;		

	elsif(rising_edge(clk))then
		stateout <= (others => '0');
		mem_data_o <= (others => '0');
		mem_wren_o <= '0';
		mem_wren_o <= '0';

		case state is
		
				when init =>
					mem_addr_o 	<= mem_addr_o + '1';
					mem_data_o	<= (others => '0');
					mem_wren_o  <= '1';
					if ( mem_addr_o = x"FFFE" ) then
						mem_addr_finished_out <= (others => '0');
						state <= waiting;
					end if;		

				when waiting =>
					stateout(3 downto 0) <= x"1";
					
					--LOOP link mux take the last one for prio
                    link_mux:
                    FOR i in 0 to NLINKS - 1 LOOP
                        if (link_data_in(7 + i * 32 downto i * 32) = x"BC" 
						and link_data_in_k(3 + i * 4 downto i * 4) = "0001" 
						and link_data_in((i + 1) * 32 - 1 downto 26 + i * 32) = "000111") then
							stateout(3 downto 0) <= x"1";
							mem_addr_o <= mem_addr_o + '1';
							mem_data_o <= link_data_in((i + 1) * 32 - 1 downto i * 32);
							mem_wren_o <= '1';
							state <= starting;
							current_link <= i;
                        end if;
                    END LOOP link_mux;

				when starting =>
					stateout(3 downto 0) <= x"2";
					if (link_data_in_k(3 + current_link * 4 downto current_link * 4) = "0000") then
						mem_addr_o <= mem_addr_o + '1';
						mem_data_o <= link_data_in((current_link + 1) * 32 - 1 downto current_link * 32);
						mem_wren_o <= '1';
					elsif (link_data_in(7 + current_link * 32 downto current_link * 32) = x"9C" and link_data_in_k(3 + current_link * 4 downto current_link * 4) = "0001") then
						mem_addr_o <= mem_addr_o + '1';
						mem_addr_finished_out <= mem_addr_o + '1';
						mem_data_o <= link_data_in((current_link + 1) * 32 - 1 downto current_link * 32);
						mem_wren_o <= '1';
						state <= waiting;
					end if;

				when others =>
					stateout(3 downto 0) <= x"E";
					mem_data_o <= (others => '0');
					mem_wren_o <= '0';
					state <= waiting;
		end case;
		
	end if;
	end process;

end architecture;
