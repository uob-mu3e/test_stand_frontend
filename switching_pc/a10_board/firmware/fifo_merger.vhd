library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
--use ieee.std_logic_unsigned.all;

entity fifo_merger is
generic (
    W : positive := 32;
    N : positive--;
);
port (
    i_rdata     : in    std_logic_vector(N*W-1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0); -- start of packet
    i_reop      : in    std_logic_vector(N-1 downto 0); -- end of packet
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0);

    o_wdata     : out   std_logic_vector(W-1 downto 0);
    o_wsop      : out   std_logic; -- start of packet
    o_weop      : out   std_logic; -- end of packet
    o_we        : out   std_logic;

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of fifo_merger is

    signal index : integer range 0 to N-1 := 0;

    function next_index (
        index : integer;
        empty : std_logic_vector--;
    ) return integer is
        variable i : integer;
    begin
        for i in index+1 to empty'length loop
            if ( empty(i) = '0' ) then
                return i;
            end if;
        end loop;
        for i in 0 to index-1 loop
            if ( empty(i) = '0' ) then
                return i;
            end if;
        end loop;
        if ( index = empty'length-1 ) then
            return 0;
        else
            return index + 1;
        end if;
    end function;

    signal rsop : std_logic_vector(N-1 downto 0);

begin

    o_rack <= (
        index => not i_rempty(index),
        others => '0'
    );

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_rack <= (others => '0');
        o_we <= '0';
        --
    elsif rising_edge(i_clk) then
        o_wdata <= i_rdata(W-1 + W*index downto 0 + W*index);
        o_wsop <= i_rsop(index);
        o_weop <= i_reop(index);
        o_we <= '0';

        if ( i_rempty(index) = '0' ) then
            -- write data
            o_we <= '1';

            if ( i_rsop(index) = '1' ) then
                -- mark SOP
                rsop <= i_rsop(index);
            end if;

            if ( i_reop(index) = '1' ) then
                -- reset SOP
                rsop <= '0';
                -- go to next index
                index <= next_index(index, i_rempty);
            end if;
        elsif ( rsop(index) = '0' ) then
            -- not inside packet -> go to next index
            index <= next_index(index, i_rempty);
        end if;

        --
    end if;
    end process;

end architecture;
