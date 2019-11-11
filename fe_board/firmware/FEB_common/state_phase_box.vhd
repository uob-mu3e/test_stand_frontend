-- sync FEB state from reset rx clk to global clock & measure phase between them 
-- Martin Mueller, March 2019 

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

ENTITY state_phase_box is
PORT (
    -- input state (rx recovered clock)
    i_state_125_rx      : in    run_state_t;
    -- _same_ frequence as global 125 clock
    i_clk_125_rx        : in    std_logic;

    -- output state (global 125 clock)
    o_state_125         : out   run_state_t;
    i_reset_125_n       : in    std_logic;
    i_clk_125           : in    std_logic;

    phase               : out   std_logic_vector(31 downto 0);
    i_reset_n           : in    std_logic;
    -- free running clock
    i_clk               : in    std_logic--;
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
    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            counter                 <= (others => '0');
            phase                   <= (others => '0');
            single_result           <= '0';
        elsif rising_edge(i_clk) then
            counter <= counter + 1;
            if(counter(26)='1') then
                counter             <= (others => '0');
                phase               <= std_logic_vector(phase_counter);
                phase_counter       <= (others => '0');
                
            -- metastable result :
            elsif ( i_clk_125_rx /= i_clk_125 ) then
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
        rst_n   => i_reset_n,
        clk     => i_clk
    );

    process(i_clk_125, i_reset_125_n)
    begin
    if ( i_reset_125_n = '0' ) then
        o_state_125 <= RUN_STATE_IDLE;
        --
    elsif rising_edge(i_clk_125) then
        if(i_state_125_rx = RUN_STATE_IDLE)        then o_state_125 <= RUN_STATE_IDLE; end if;
        if(i_state_125_rx = RUN_STATE_PREP)        then o_state_125 <= RUN_STATE_PREP; end if;
        if(i_state_125_rx = RUN_STATE_SYNC)        then o_state_125 <= RUN_STATE_SYNC; end if;
        if(i_state_125_rx = RUN_STATE_RUNNING)     then o_state_125 <= RUN_STATE_RUNNING; end if;
        if(i_state_125_rx = RUN_STATE_TERMINATING) then o_state_125 <= RUN_STATE_TERMINATING; end if;
        if(i_state_125_rx = RUN_STATE_LINK_TEST)   then o_state_125 <= RUN_STATE_LINK_TEST; end if;
        if(i_state_125_rx = RUN_STATE_SYNC_TEST)   then o_state_125 <= RUN_STATE_SYNC_TEST; end if;
        if(i_state_125_rx = RUN_STATE_RESET)       then o_state_125 <= RUN_STATE_RESET; end if;
        if(i_state_125_rx = RUN_STATE_OUT_OF_DAQ)  then o_state_125 <= RUN_STATE_OUT_OF_DAQ; end if;
        if(i_state_125_rx = RUN_STATE_IDLE)        then o_state_125 <= RUN_STATE_IDLE; end if;
    end if;
    end process;

END architecture;
