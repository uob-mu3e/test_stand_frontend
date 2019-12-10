library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_fifo_sc is
end entity;

architecture arch of tb_fifo_sc is

    signal clk, reset_n : std_logic := '0';

    signal wd, rd : std_logic_vector(3 downto 0) := (others => '0');
    signal wfull, rempty, we, re : std_logic := '0';

begin

    clk <= not clk after 5 ns;
    reset_n <= '0', '1' after 100 ns;

    e_fifo : entity work.fifo_sc
    generic map (
        DATA_WIDTH_g => wd'length,
        ADDR_WIDTH_g => 4--,
    )
    port map (
        o_wfull     => wfull,
        i_we        => we,
        i_wdata     => wd,

        o_rempty    => rempty,
        i_re        => re,
        o_rdata     => rd,

        i_reset_n   => reset_n,
        i_clk       => clk--,
    );

    process
        variable i : unsigned(wd'range) := (others => '0');
    begin
        we <= '0';
        if ( wfull = '0' ) then
            we <= '1';
            wd <= std_logic_vector(i);
            report "WRITE: i = " & work.util.to_string(i) & ", wd = " & work.util.to_string(i);
            i := i + 1;
        end if;
        wait until rising_edge(clk);
    end process;

    process
        variable i : unsigned(rd'range) := (others => '0');
    begin
        if ( rempty = '0' ) then
            report "READ: i = " & work.util.to_string(i) & ", rd = " & work.util.to_string(rd);
            assert ( rd = std_logic_vector(i) ) report "ERROR" severity error;
            i := i + 1;
        end if;
        wait until rising_edge(clk);
    end process;
    
    re <= not rempty;

end architecture;
