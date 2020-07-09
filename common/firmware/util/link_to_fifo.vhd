library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity link_to_fifo is
generic (
    W : positive := 32--;
);
port (

    i_link_data  : in std_logic_vector(W-1 downto 0);
    i_link_datak : in std_logic_vector(3 downto 0);
    i_fifo_almost_full : in std_logic;    

    o_fifo_data  : out std_logic_vector(W + 3 downto 0);
    o_fifo_wr    : out std_logic;

    i_reset_n : in std_logic;
    i_clk     : in std_logic--;
);
end entity;

architecture arch of link_to_fifo is
   
    type link_to_fifo_type is (idle, write_data, skip_data);
    signal link_to_fifo_state : link_to_fifo_type;

begin

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        o_fifo_data <= (others => '0');
        o_fifo_wr <= '0';
        link_to_fifo_state <= idle;
        --
    elsif rising_edge(i_clk) then

        o_fifo_wr <= '0';
        o_fifo_data <= i_link_data & i_link_datak;
        
        case link_to_fifo_state is 

        when idle =>
            if ( i_link_data = x"000000BC" and i_link_datak = "0001" ) then
                --
            elsif ( i_link_data(7 downto 0) = x"BC" and i_link_datak = "0001" ) then
                if ( i_fifo_almost_full = '1' ) then
                    link_to_fifo_state <= skip_data;
                else
                    link_to_fifo_state <= write_data;
                    o_fifo_wr <= '1';
                end if;
            end if;

        when write_data =>
            if ( i_link_data(7 downto 0) = x"9C" and i_link_datak = "0001" ) then
                link_to_fifo_state <= idle;
            end if;

            if ( i_link_data = x"000000BC" and i_link_datak = "0001" ) then
                --
            else
                o_fifo_wr <= '1';
            end if;

        when skip_data =>
            if ( i_link_data(7 downto 0) = x"9C" and i_link_datak = "0001" ) then
                link_to_fifo_state <= idle;
            end if;

        when others =>
            link_to_fifo_state <= idle;
        
        end case;
    --
    end if;
    end process;

end architecture;
