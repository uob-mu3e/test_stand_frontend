-------------------------------------------------------
--! @farm_link_to_fifo.vhd
--! @brief the farm_link_to_fifo sorts out the data from the
--! link and provides it as a fifo output
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity farm_link_to_fifo is
generic (
    g_NLINKS_SWB_TOTL    : positive :=  16;
    N_PIXEL              : positive :=  8;
    N_SCIFI              : positive :=  8;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in  work.mu3e.link_array_t(g_NLINKS_SWB_TOTL-1 downto 0);

    -- pixel data
    o_pixel         : out std_logic_vector(N_PIXEL * 32 + 1 downto 0);
    o_empty_pixel   : out std_logic;
    i_ren_pixel     : in  std_logic;
    o_error_pixel   : out std_logic;

    -- scifi data
    o_scifi         : out std_logic_vector(N_SCIFI * 32 + 1 downto 0);
    o_empty_scifi   : out std_logic;
    i_ren_scifi     : in  std_logic;
    o_error_scifi   : out std_logic;

    --! error counters
    --! 0: fifo sync_almost_full (pixel)
    --! 1: fifo sync_wrfull (pixel)
    --! 2: # of overflow event (pixel)
    --! 3: cnt events (pixel)
    --! 4: fifo sync_almost_full (scifi)
    --! 5: fifo sync_wrfull (scifi)
    --! 6: # of overflow event (scifi)
    --! 7: cnt events (scifi)
    o_counter       : out work.util.slv32_array_t(7 downto 0);

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of farm_link_to_fifo is

    constant check_ones   : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0) := (others => '1');
    constant check_zeros  : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0) := (others => '0');

    type link_to_fifo_type is (idle, write_data, skip_data);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_event : std_logic_vector(31 downto 0);

    signal rx_data, rx_q : work.util.slv34_array_t(g_NLINKS_SWB_TOTL - 1 downto 0);
    signal rx_wen, sync_rdempty, sync_ren, sop, eop : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0);

    signal rx_pixel : work.util.slv34_array_t(N_PIXEL - 1 downto 0);
    signal rx_scifi : work.util.slv34_array_t(N_SCIFI - 1 downto 0);

    signal f_data, f_q : std_logic_vector(g_NLINKS_SWB_TOTL * 36 - 1 downto 0);
    signal f_almost_full, f_wrfull, f_wen : std_logic;
    signal f_wrusedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH - 1 downto 0);
    signal counter_pixel, counter_scifi : work.util.slv32_array_t(3 downto 0);

begin

    --! sync link data from link to pcie clk
    gen_link_to_fifo : FOR i in 0 to g_NLINKS_SWB_TOTL - 1 GENERATE

        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n = '0' ) then
            rx_data(i)  <= (others => '0');
            rx_wen(i)   <= '0';
        elsif rising_edge(i_clk) then
            -- idle word
            if ( i_rx(i).data = x"000000BC" and i_rx(i).datak = "0001" ) then
                rx_wen(i) <= '0';
            -- header
            elsif ( i_rx(i).data(7 downto 0) = x"7C" and i_rx(i).datak = "0001" ) then
                rx_data(i) <= "01" & i_rx(i).data;
                rx_wen(i) <= '1';
            -- trailer
            elsif ( i_rx(i).data(7 downto 0) = x"9C" and i_rx(i).datak = "0001" ) then
                rx_data(i) <= "10" & i_rx(i).data;
                rx_wen(i) <= '1';
            -- hits
            else
                rx_data(i) <= "00" & i_rx(i).data;
                rx_wen(i) <= '1';
            end if;
        end if;
        end process;

        e_sync_fifo : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => 4,
            DATA_WIDTH  => 34--,
        )
        port map (
            data        => rx_data(i),
            wrreq       => rx_wen(i),
            rdreq       => sync_ren(i),
            wrclk       => i_clk,
            rdclk       => i_clk,
            q           => rx_q(i),
            rdempty     => sync_rdempty(i),
            aclr        => not i_reset_n--,
        );

        sop(i) <= '1' when rx_q(i)(33 downto 32) = "01" else '0';
        eop(i) <= '1' when rx_q(i)(33 downto 32) = "10" else '0';

    END GENERATE;

    gen_counter : FOR I in 0 to 3 GENERATE
        o_counter(I) <= counter_pixel(I);
        o_counter(I+4) <= counter_scifi(I);
    END GENERATE;

    gen_map_pixel : FOR I in 0 to N_PIXEL - 1 GENERATE
        rx_pixel(I) <= rx_q(I);
    END GENERATE;

    gen_map_scifi : FOR I in 0 to N_SCIFI - 1 GENERATE
        rx_scifi(I) <= rx_q(I+N_PIXEL);
    END GENERATE;


    e_aligne_pixel : entity work.farm_aligne_link
    generic map (
        N => N_PIXEL,
        LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
    )
    port map (
        i_rx    => rx_pixel,
        i_sop   => sop(N_PIXEL - 1 downto 0),
        i_eop   => eop(N_PIXEL - 1 downto 0),

        --! error counters
        --! 0: fifo sync_almost_full
        --! 1: fifo sync_wrfull
        --! 2: # of overflow event
        --! 3: cnt events
        o_counter   => counter_pixel,
        o_data      => o_pixel,
        o_empty     => o_empty_pixel,
        i_ren       => i_ren_pixel,

        i_empty     => sync_rdempty(N_PIXEL - 1 downto 0),
        o_ren       => sync_ren(N_PIXEL - 1 downto 0),

        o_error     => o_error_pixel,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    e_aligne_scifi : entity work.farm_aligne_link
    generic map (
        N => N_SCIFI,
        LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
    )
    port map (
        i_rx    => rx_scifi,
        i_sop   => sop(N_SCIFI + N_PIXEL - 1 downto N_PIXEL),
        i_eop   => eop(N_SCIFI + N_PIXEL - 1 downto N_PIXEL),

        --! error counters
        --! 0: fifo sync_almost_full
        --! 1: fifo sync_wrfull
        --! 2: # of overflow event
        --! 3: cnt events
        o_counter   => counter_scifi,
        o_data      => o_scifi,
        o_empty     => o_empty_scifi,
        i_ren       => i_ren_scifi,

        i_empty     => sync_rdempty(N_SCIFI + N_PIXEL - 1 downto N_PIXEL),
        o_ren       => sync_ren(N_SCIFI + N_PIXEL - 1 downto N_PIXEL),

        o_error     => o_error_scifi,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

end architecture;
