-- sync FEB state from reset rx clk to global clock & measure phase between them 
-- Martin Mueller, March 2019 

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

ENTITY state_phase_box is
PORT (
    clk_global              : in    std_logic;
    clk_rx_reset            : in    std_logic;
    clk_free                : in    std_logic;
    reset                   : in    std_logic; 
    phase                   : out   std_logic_vector(31 downto 0);
    -- states in sync to clk_rx_reset:
    state_idle_rx           : in    std_logic;
    state_run_prepare_rx    : in    std_logic;
    state_sync_rx           : in    std_logic;
    state_running_rx        : in    std_logic;
    state_terminating_rx    : in    std_logic;
    state_link_test_rx      : in    std_logic;
    state_sync_test_rx      : in    std_logic;
    state_reset_rx          : in    std_logic;
    state_out_of_DAQ_rx     : in    std_logic;
    -- states in sync to clk_global:
    state_sync_global       : out   feb_run_state
);
END ENTITY;

architecture rtl of state_phase_box is

	signal counter : unsigned(31 downto 0);
	signal delay: std_logic;
	signal phase_counter : unsigned(31 downto 0);

begin

	-- measure phase between clk_reset and clk_global
	process (clk_free,reset)
	begin
		if reset = '1'  then 
			counter		<= (others => '0');
			phase			<= (others => '0');
		elsif rising_edge(clk_free) then
			counter <= counter + 1;
			if(counter(26)='1') then
				counter				<= (others => '0');
				phase					<= std_logic_vector(phase_counter);
				phase_counter		<= (others => '0');
			elsif(clk_global /= clk_rx_reset) then
				phase_counter <= phase_counter + 1;
			end if;
		end if;
	end process;


	process (clk_global, reset)
	begin
		if reset = '1' then
			state_sync_global		<= idle;
		elsif rising_edge(clk_global) then
			if(state_running_rx = '1') 	then state_sync_global <= running; end if;
			if(state_idle_rx = '1') 		then state_sync_global <= idle; end if;
			if(state_run_prepare_rx = '1') then state_sync_global <= run_prep; end if;
			if(state_sync_rx = '1') then state_sync_global <= sync; end if;
			if(state_terminating_rx = '1') then state_sync_global <= terminating; end if;
			if(state_reset_rx = '1') then state_sync_global <= reset_state; end if;
			if(state_link_test_rx = '1') then state_sync_global <= link_test; end if;
			if(state_sync_test_rx = '1') then state_sync_global <= sync_test; end if;
			if(state_out_of_DAQ_rx = '1') then state_sync_global <= out_of_DAQ; end if;
		end if;
	end process;

END architecture;
