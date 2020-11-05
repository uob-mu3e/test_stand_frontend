
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.vc_context;

entity counter_tb is
    generic (runner_cfg : string);
end entity;

architecture arch of counter_tb is

    constant CLK_MHZ : real := 1000.0; -- MHz
    signal clk, reset_n, valid : std_logic := '0';
    
    constant W : positive := 32;
    signal cnt : std_logic_vector(W-1 downto 0);
    
begin

    clk <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);
    
    e_counter : entity work.counter
    generic map (
        W => W--,
    )
    port map (
        o_cnt       => cnt,
        i_ena       => '1',

        i_reset_n   => reset_n,
        i_clk       => clk--,
    );

    valid <= '1' when cnt < x"00000010" else '0';
    
    test_p : process
    begin
        test_runner_setup(runner, runner_cfg);
        
        while test_suite loop
            reset_checker_stat;
            if run("test_counter_zero_at_start") then
                wait until valid = '1' for 100 ns;
                check_equal(valid, '0');
            elsif run("test_counter_values") then
                wait until rising_edge(clk);
                check_equal(valid, '1');
            end if;
        end loop;

        test_runner_cleanup(runner);
        wait;
    end process;
    test_runner_watchdog(runner, 10 ms); 

end architecture;
