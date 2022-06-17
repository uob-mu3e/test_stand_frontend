library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;

-- merge packets delimited by SOP and EOP from N input streams
entity stream_merger_vector is
generic (
    g_set_type : boolean := false;
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    work.util.slv34_array_t(N-1 downto 0);
    i_sop       : in    std_logic_vector(N-1 downto 0);
    i_eop       : in    std_logic_vector(N-1 downto 0);
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0); -- read ACK

    -- output stream
    o_wdata     : out   std_logic_vector(33 downto 0);
    i_wfull     : in    std_logic;
    o_busy      : out   std_logic;
    o_we        : out   std_logic; -- write enable

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of stream_merger_vector is

    -- one hot signal indicating current active input stream
    signal index : std_logic_vector(N-1 downto 0) := (0 => '1', others => '0');

    -- SOP mark
    signal busy : std_logic;

begin

    -- set rack for current not empty input
    o_rack <= (others => '0') when ( i_wfull = '1' or i_reset_n = '0' ) else
        not i_rempty and index;

    o_busy <= busy;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        o_wdata <= (others => '0');
        o_we    <= '0';

        index   <= (0 => '1', others => '0');
        busy    <= '0';
        --
    elsif rising_edge(i_clk) then
        for i in N-1 downto 0 loop
            if ( index(i) = '1' ) then
                o_wdata <= i_rdata(i);
                if ( i_sop(i) = '1' and g_set_type ) then
                    o_wdata(25 downto 20) <= work.mudaq.link_36_to_std(i);
                end if;
            end if;
        end loop;
        o_we <= '0';

        if ( work.util.or_reduce(i_rempty and index) = '0' and i_wfull = '0' ) then
            -- write data
            o_we <= '1';

            if ( work.util.or_reduce(i_sop and index) = '1' ) then
                -- set SOP mark
                busy <= '1';
            end if;

            if ( work.util.or_reduce(i_eop and index) = '1' ) then
                -- reset SOP mark
                busy <= '0';
                -- go to next index
                index <= work.util.round_robin_next(index, not i_rempty);
            end if;
        elsif ( busy = '0' ) then
            -- go to next index
            index <= work.util.round_robin_next(index, not i_rempty);
        end if;

        --
    end if;
    end process;

end architecture;
