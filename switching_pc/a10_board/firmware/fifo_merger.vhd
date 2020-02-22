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
    i_rsop      : in    std_logic_vector(N-1 downto 0); -- start of packet (SOP)
    i_reop      : in    std_logic_vector(N-1 downto 0); -- end of packet (EOP)
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0);

    o_wdata     : out   std_logic_vector(W-1 downto 0);
    o_wsop      : out   std_logic; -- SOP
    o_weop      : out   std_logic; -- EOP
    o_we        : out   std_logic;

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of fifo_merger is

    -- current index
    signal index : integer range 0 to N-1 := 0;

    -- get next index such that empty(index) = '0'
    function next_index (
        index : integer;
        empty : std_logic_vector--;
    ) return integer is
    begin
        for i in empty'range loop
            index := (index + 1) mod empty'length;
            exit when ( empty(index) = '0' );
        end loop;
        return index;
    end function;

    -- SOP marker
    signal sop : std_logic_vector(N-1 downto 0);

begin

    -- drive rack (read ack) for current not empty input
    process(index, i_rempty, i_reset_n)
    begin
        o_rack <= (others => '0');
        if ( i_rempty(index) = '0' and i_reset_n = '1' ) then
            o_rack(index) <= '1';
        end if;
    end process;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        o_wdata <= (others => '0');
        o_wsop <= (others => '0');
        o_weop <= (others => '0');
        o_we <= '0';

        index <= 0;
        sop <= (others => '0');
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
                sop(index) <= i_rsop(index);
            end if;

            if ( i_reop(index) = '1' ) then
                -- reset SOP
                sop(index) <= '0';
                -- go to next index
                index <= next_index(index, i_rempty);
            end if;
        elsif ( sop(index) = '0' ) then
            -- not inside packet -> go to next index
            index <= next_index(index, i_rempty);
        end if;

        --
    end if;
    end process;

end architecture;
