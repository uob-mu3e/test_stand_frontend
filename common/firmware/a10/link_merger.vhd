-----------------------------------------------------------------------------
-- Merging links for the farm PCs
--
-- Marius Koeppel, JGU Mainz
-- mkoeppel@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity link_merger is
generic (
    W : integer := 66;
    NLINKS_TOTL : integer := 3;
    TREE_DEPTH_w : positive := 8;
    TREE_DEPTH_r : positive := 8;
    LINK_FIFO_ADDR_WIDTH : integer := 10--;
);
port (
    i_reset_data_n : in std_logic;
    i_reset_mem_n : in std_logic;
    i_dataclk : in std_logic;
    i_memclk : in std_logic;

    i_link              : in    work.mu3e.link_array_t(NLINKS_TOTL-1 downto 0);
    i_link_valid : in integer;
    i_link_mask_n : in std_logic_vector(NLINKS_TOTL - 1 downto 0);

    o_stream_rdata : out std_logic_vector(W - 1 downto 0); -- "11" = shop, "10" = eop, "01" = sop, "00" = data
    o_stream_rempty : out std_logic;
    i_stream_rack : in std_logic--;

);
end entity;

architecture arch of link_merger is

    signal reset_data, reset_mem : std_logic;

    signal link_data, link_dataq : work.mu3e.link_array_t(NLINKS_TOTL-1 downto 0);
    signal link_empty, link_wren, link_full, link_afull, link_wrfull, sop, eop, shop, link_ren : std_logic_vector(NLINKS_TOTL - 1 downto 0);
    signal link_usedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH * NLINKS_TOTL - 1 downto 0);
    signal sync_fifo_empty : std_logic_vector(NLINKS_TOTL - 1 downto 0);
    signal sync_fifo_i_wrreq : std_logic_vector(NLINKS_TOTL - 1 downto 0);
    signal sync_fifo_q : work.mu3e.link_array_t(NLINKS_TOTL-1 downto 0);
    signal sync_fifo_data : work.mu3e.link_array_t(NLINKS_TOTL-1 downto 0);

    signal stream_wdata, stream_rdata : std_logic_vector(W-1 downto 0);
    signal we_counter : std_logic_vector(63 downto 0);
    signal stream_rempty, stream_rack, stream_wfull, stream_we : std_logic;
    signal hit_a : work.util.slv32_array_t(7 downto 0);

begin

    reset_data <= not i_reset_data_n;
    reset_mem <= not i_reset_mem_n;

    buffer_link_fifos: FOR i in 0 to NLINKS_TOTL - 1 GENERATE

        process(i_dataclk, i_reset_data_n)
        begin
        if ( i_reset_data_n = '0' ) then
            sync_fifo_data(i) <= work.mu3e.LINK_ZERO;
            sync_fifo_i_wrreq(i) <= '0';
        elsif rising_edge(i_dataclk) then
            sync_fifo_data(i) <= i_link(i);
            if ( i_link(i).idle = '1' ) then
                sync_fifo_i_wrreq(i) <= '0';
            else
                sync_fifo_i_wrreq(i) <= '1';
            end if;
        end if;
        end process;

        e_sync_fifo : entity work.link_dcfifo
        generic map(
            g_ADDR_WIDTH  => 6--,
        )
        port map (
            i_we        => sync_fifo_i_wrreq(i),
            i_wdata     => sync_fifo_data(i),
            i_wclk      => i_dataclk,

            i_rack      => not sync_fifo_empty(i),
            o_rdata     => sync_fifo_q(i),
            o_rempty    => sync_fifo_empty(i),
            i_rclk      => i_memclk,

            i_reset_n   => '1'--,
        );

        e_link_to_fifo : entity work.link_to_fifo
        port map (
            i_link              => sync_fifo_q(i),
            i_fifo_almost_full  => link_afull(i),
            i_sync_fifo_empty   => sync_fifo_empty(i),
            o_fifo_data         => link_data(i),
            o_fifo_wr           => link_wren(i),
            o_cnt_skip_data     => open,
            i_reset_n           => i_reset_mem_n,
            i_clk               => i_memclk--,
        );

        e_fifo : entity work.link_dcfifo
        generic map (
            g_ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_we        => link_wren(i),
            i_wdata     => link_data(i),
            o_wusedw    => link_usedw(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1 downto i * LINK_FIFO_ADDR_WIDTH),
            i_wclk      => i_dataclk,

            i_rack      => link_ren(i),
            o_rdata     => link_dataq(i),
            o_rempty    => link_empty(i),
            i_rclk      => i_memclk,

            i_reset_n   => not reset_data--,
        );

        process(i_dataclk, i_reset_data_n)
        begin
        if ( i_reset_data_n = '0' ) then
            link_afull(i)       <= '0';
        elsif rising_edge(i_dataclk) then
            if(link_usedw(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1) = '1') then
                link_afull(i)   <= '1';
            else
                link_afull(i)   <= '0';
            end if;
        end if;
        end process;

        sop(i) <= link_dataq(i).sop;
        shop(i) <= '1' when link_dataq(i).eop = '0' and link_dataq(i).sop = '0' and link_dataq(I).data(27 downto 22) = "111111" else '0';
        eop(i) <= link_dataq(i).eop;

    END GENERATE buffer_link_fifos;

    e_time_merger : entity work.time_merger
        generic map (
        W => W,
        TREE_DEPTH_w  => TREE_DEPTH_w,
        TREE_DEPTH_r  => TREE_DEPTH_r,
        N => NLINKS_TOTL--,
    )
    port map (
        -- input streams
        i_rdata                 => link_dataq,
        i_rshop                 => shop,
        i_rempty                => link_empty,
        i_link                  => i_link_valid,
        i_mask_n                => i_link_mask_n,
        o_rack                  => link_ren,

        -- output stream
        o_rdata                 => stream_rdata,
        i_ren                   => stream_rack,
        o_empty                 => stream_rempty,

        -- error outputs

        i_reset_n               => i_reset_mem_n,
        i_clk                   => i_memclk--,
    );

    process(i_memclk, i_reset_mem_n)
    begin
    if ( i_reset_mem_n /= '1' ) then
        we_counter <= (others => '0');
    elsif rising_edge(i_memclk) then
        if ( stream_we = '1' ) then
            we_counter <= we_counter + '1';
        end if;
    end if;
    end process;

    o_stream_rdata <= stream_rdata;
    o_stream_rempty <= stream_rempty;
    stream_rack <= i_stream_rack;

end architecture;
