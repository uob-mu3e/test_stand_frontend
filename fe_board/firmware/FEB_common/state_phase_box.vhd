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
    state_sync_global       : out   run_state_t
);
END ENTITY;

architecture rtl of state_phase_box is

    signal counter :                unsigned(31 downto 0);
    signal delay :                  std_logic;
    signal phase_counter :          unsigned(31 downto 0);
    signal single_result :          std_logic;
    signal single_result_stable :   std_logic;

begin

    -- measure phase between clk_reset and clk_global
    process (clk_free,reset)
    begin
        if reset = '1'  then 
            counter                 <= (others => '0');
            phase                   <= (others => '0');
            single_result           <= '0';
        elsif rising_edge(clk_free) then
            counter <= counter + 1;
            if(counter(26)='1') then
                counter             <= (others => '0');
                phase               <= std_logic_vector(phase_counter);
                phase_counter       <= (others => '0');
                
            -- metastable result :
            elsif(clk_global /= clk_rx_reset) then
                single_result       <= '1';
            else
                single_result       <= '0';
            end if;
            
            -- count phase with stable result :
            if (single_result_stable = '1') then
                phase_counter <= phase_counter + 1;
            end if;
        end if;
    end process;
    
    -- sync metastable result
    i_ff_sync : entity work.ff_sync
    generic map ( W => 1, N => 5 )
    PORT MAP (
        d(0)    => single_result,
        q(0)    => single_result_stable,
        rst_n   => not reset,
        clk     => clk_free
    );


	process (clk_global, reset)
	begin
		if reset = '1' then
			state_sync_global		<= RUN_STATE_IDLE;
		elsif rising_edge(clk_global) then
			if(state_running_rx = '1')     then state_sync_global <= RUN_STATE_RUNNING; end if;
			if(state_idle_rx = '1')        then state_sync_global <= RUN_STATE_IDLE; end if;
			if(state_run_prepare_rx = '1') then state_sync_global <= RUN_STATE_PREP; end if;
			if(state_sync_rx = '1')        then state_sync_global <= RUN_STATE_SYNC; end if;
			if(state_terminating_rx = '1') then state_sync_global <= RUN_STATE_TERMINATING; end if;
			if(state_reset_rx = '1')       then state_sync_global <= RUN_STATE_RESET; end if;
			if(state_link_test_rx = '1')   then state_sync_global <= RUN_STATE_LINK_TEST; end if;
			if(state_sync_test_rx = '1')   then state_sync_global <= RUN_STATE_SYNC_TEST; end if;
			if(state_out_of_DAQ_rx = '1')  then state_sync_global <= RUN_STATE_OUT_OF_DAQ; end if;
		end if;
	end process;

END architecture;
