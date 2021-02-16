library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity lapse_counter_tb is
end entity;

architecture arch of lapse_counter_tb is

    constant CLK_MHZ : real := 1000.0; -- MHz
    constant N_CC : integer := 15;
    signal clk, reset_n, reset : std_logic := '0';

    signal DONE : std_logic_vector(1 downto 0) := (others => '0');

    signal i_CC : std_logic_vector(N_CC - 1 downto 0);
    signal o_CC : std_logic_vector(N_CC downto 0);

begin

    clk <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);
    reset <= not reset_n;

    p_gen_cc: process(clk, reset_n)
    begin
        if ( reset_n = '0' ) then
            i_CC <= (others => '0');
        elsif ( rising_edge(clk) ) then
            if ( i_CC = 32767 - 1 ) then
                i_CC <= (others => '0');
            else
                i_CC <= i_CC + 1;
            end if;
        end if;
    end process;

    e_lapse_counter : entity work.lapse_counter
    generic map (N_TOT => 32767, N_CC => N_CC) 
    port map ( i_clk => clk, i_reset_n => reset_n, i_CC => i_CC, o_CC => o_CC );

end architecture;
