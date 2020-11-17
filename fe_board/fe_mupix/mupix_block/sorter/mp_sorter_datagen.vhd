library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mp_sorter_datagen is
port (
    reset_n         : in  std_logic;
    clk             : in  std_logic;
    running         : in  std_logic;
    enable          : in  std_logic;
    fifo_wdata      : out std_logic_vector(35 downto 0);
    fifo_write      : out std_logic--;
);
end entity;

architecture rtl of mp_sorter_datagen is

    signal wdata_b : std_logic_vector(35 downto 0);

begin
    fifo_wdata <= wdata_b;
    
    process(clk,reset_n)
    begin
        if ( reset_n = '0' ) then
            wdata_b  <= (others => '0');
            fifo_write  <= '0';

        elsif rising_edge(clk) then
            wdata_b <= not wdata_b;
            
        end if;
    end process;

end architecture;