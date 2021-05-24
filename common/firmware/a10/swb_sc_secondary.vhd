----------------------------------------------------------------------------
-- Slow Control Secondary Unit for Switching Board
-- Marius Koeppel, Mainz University
-- mkoeppel@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


entity swb_sc_secondary is
generic (
    NLINKS : integer := 4;
		skip_init : std_logic := '0'
);
port (
		clk:                in std_logic;
		reset_n:            in std_logic;
    i_link_enable               : in    std_logic_vector(NLINKS-1 downto 0);
    link_data_in                : in    work.util.slv32_array_t(NLINKS-1 downto 0);
    link_data_in_k              : in    work.util.slv4_array_t(NLINKS-1 downto 0);
		mem_data_out:       out std_logic_vector(31 downto 0);
		mem_addr_out:       out std_logic_vector(15 downto 0);
		mem_addr_finished_out:       out std_logic_vector(15 downto 0);
		mem_wren:           out std_logic;
		stateout:           out std_logic_vector(3 downto 0)
);
end entity;

architecture arch of swb_sc_secondary is

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
		if (skip_init = '0') then
			state <= init;
		else
	  		state <= waiting;
	  	end if;

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
						if (  i_link_enable(i)='1'
							and link_data_in(i)(7 downto 0) = x"BC"
							and link_data_in_k(i) = "0001"
							and link_data_in(i)(31 downto 26) = "000111"
						) then
							stateout(3 downto 0) <= x"1";
							mem_addr_o <= mem_addr_o + '1';
							mem_data_o <= link_data_in(i);
							mem_wren_o <= '1';
							state <= starting;
							current_link <= i;
							end if;
					END LOOP;

				when starting =>
					stateout(3 downto 0) <= x"2";
					if (link_data_in_k(current_link) = "0000") then
						mem_addr_o <= mem_addr_o + '1';
						mem_data_o <= link_data_in(current_link);
						mem_wren_o <= '1';
					elsif (link_data_in(current_link)(7 downto 0) = x"9C" and link_data_in_k(current_link) = "0001") then
						mem_addr_o <= mem_addr_o + '1';
						mem_addr_finished_out <= mem_addr_o + '1';
						mem_data_o <= link_data_in(current_link);
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
