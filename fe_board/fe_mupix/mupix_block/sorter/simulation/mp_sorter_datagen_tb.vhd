library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mp_sorter_datagen_tb is
end entity;

architecture rtl of mp_sorter_datagen_tb is

    constant CLK_MHZ : positive := 125;
    signal clk, reset_n : std_logic := '0';
    signal reset : std_logic;


    signal fifo_wdata : std_logic_vector(35 downto 0);
    signal fifo_write : std_logic;

begin

    clk <= not clk after (500 ns / CLK_MHZ);
    reset_n <= '0', '1' after 32 ns;
    reset <= not reset_n;

    e_mp_sorter_datagen : entity work.mp_sorter_datagen
    port map (
        reset_n         => reset_n,
        clk             => clk,
        running         => '1',
        enable          => '1',
        fifo_wdata      => fifo_wdata,
        fifo_write      => fifo_write--,
    );

    process
    begin
        wait until ( reset_n = '1' );


        wait;
    end process;

end architecture;