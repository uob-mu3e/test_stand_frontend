--**** selection unit interfaces with roundrobin_alternant architecture
-- Author: K.Briggl, Dec. 2015
-- Version 0.02: by Huangshan Chen May 2016
--		change entity name to arb_selection_alter to distiguish from arb_selection for dc_shell
--		to use roundrobin_sel_alternant architecture
-- K. Briggl (Now in geneva), May 2019
--		Modifications to use in daq firmware. Prioritize last two inputs, they serve as inputs for the common header and trailer
--

Library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.STD_LOGIC_ARITH.all;
use IEEE.STD_LOGIC_UNSIGNED.all;

use ieee.math_real.log2;
use ieee.math_real.ceil;


entity arb_selection_framecollect is
	generic(
		NPORTS : natural :=4 -- number of requesting ports
		);
	port(
			i_clk	: in std_logic;
			i_rst	: in std_logic;
			-- REQUEST SIGNALS
			i_req	: in std_logic_vector(NPORTS - 1 downto 0);
			-- GRANT VALID FLAG
			o_gnt_valid : out std_logic;
			i_ack 	: in std_logic;
			-- GRANT SELECTION OUTPUT
			o_gnt: out natural range NPORTS- 1 downto 0
	    );
end arb_selection_framecollect;


architecture roundrobin_plus_prio of arb_selection_framecollect is
	signal last_ack_selection, last_selection, new_selection : natural range NPORTS-1 downto 0;
	signal last_selection_valid : std_logic;
	signal p_any_req, n_any_req : std_logic;
	signal p_same_sel, n_same_sel : std_logic;
begin
	calc_next_sel : process(i_req, last_ack_selection, p_same_sel, p_any_req) --{{{
	variable this : natural range NPORTS - 1 downto 0;
	variable candidate : natural range NPORTS - 1 downto 0;
	variable selection : natural range NPORTS - 1 downto 0;
	begin
		n_any_req <='0';
		--check candidates for a request: if current is N, N+1 has priority.
		this := last_ack_selection;
		selection :=last_selection;
		for j in 0 to NPORTS - 1 loop
			candidate := (NPORTS+this-j) mod NPORTS;
			if (i_req(candidate) = '1') then
				selection := candidate;
				n_any_req<='1';
			end if;
		end loop;
		if (last_selection_valid='1' and last_selection /= last_ack_selection) then
			selection :=last_selection;
		end if;
		new_selection <=selection;

		n_same_sel <= '0';
		if selection = this and p_same_sel = '0' and p_any_req = '1' then
			n_same_sel <= '1';
		end if;
	end process;	--}}}

	sync_fsm : process(i_clk)
	begin
		if rising_edge(i_clk) then
			if i_rst = '1' then
				last_ack_selection	<= 0;
				p_same_sel	<= '0';
				p_any_req	<= '0';
			else
				p_same_sel	<= n_same_sel;
				last_selection	<= new_selection;
				last_selection_valid <= n_any_req and (not n_same_sel);
				if(i_ack = '1' and n_any_req='1') then
					last_ack_selection	<= new_selection;
					p_any_req	<= n_any_req;
				end if;
			end if;
		end if;
	end process;

	o_gnt_valid <= n_any_req and (not n_same_sel);
	o_gnt <= new_selection;
end;

