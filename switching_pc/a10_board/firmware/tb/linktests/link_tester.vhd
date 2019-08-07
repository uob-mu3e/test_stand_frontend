library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity link_tester is
    port (
        cnt     :   out std_logic_vector(31 downto 0);
        datak   :   out std_logic_vector(3 downto 0);
        reset_n :   in  std_logic;
        enable  :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity link_tester;

architecture arch of link_tester is

    signal local_counter    : std_logic_vector(31 downto 0);
    signal tmp_counter      : std_logic_vector(31 downto 0);
    signal local_datak      : std_logic_vector(3 downto 0);

begin

    cnt     <= local_counter;
    datak   <= local_datak;

    process(clk, reset_n)
    begin
    if (reset_n = '0') then
        local_counter   <= x"000000BC";
        tmp_counter     <= (others => '0');
        local_datak     <= "0001";
    elsif rising_edge(clk) then
        if (enable = '1') then
            tmp_counter     <= tmp_counter + '1';
            local_counter   <= tmp_counter;
            local_datak     <= "0000";
        else
            local_counter   <= x"000000BC";
            local_datak     <= "0001";
        end if;                
    end if;
    end process;

end architecture;