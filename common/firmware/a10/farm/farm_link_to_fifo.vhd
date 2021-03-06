-------------------------------------------------------
--! We always read from the link fifo into a fifo (link to fifo)
--! (if possible), while we tag the processed data for the 
--! next farm (farm aligne link). We align by the event #.
--! 
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
    g_LOOPUP_NAME        : string   := "intRun2021";
    g_NLINKS_SWB_TOTL    : positive :=  3;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    -- link data
    i_rx            : in  work.mu3e.link_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    o_tx            : out work.mu3e.link_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    i_mask_n        : in  std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);

    -- data out for farm path
    o_data          : out work.mu3e.link_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    o_empty         : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    i_ren           : in  std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    
    --! status counters 
    --! (g_NLINKS_DATA*5)-1 downto 0 -> link to fifo counters
    --! (g_NLINKS_DATA*4)+(g_NLINKS_DATA*5)-1 downto (g_NLINKS_DATA*5) -> link align counters
    o_counter       : out work.util.slv32_array_t((g_NLINKS_SWB_TOTL*4)+(g_NLINKS_SWB_TOTL*5)-1 downto 0);

    i_clk           : in std_logic;
    i_reset_n       : in std_logic--;
);
end entity;

architecture arch of farm_link_to_fifo is

    signal rx_q, data : work.util.slv35_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_ren, rx_mask_n, rx_rdempty : std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0) := (others => '0');
    signal sop, eop, skip : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0);

begin

    --! sync link data from link to pcie clk
    gen_link_to_fifo : FOR i in 0 to g_NLINKS_SWB_TOTL - 1 GENERATE

        -- NOTE: quick fix to send link to the next farm
        o_tx(i) <= i_rx(i) when i_mask_n(i) = '0' else work.mu3e.LINK_IDLE;
    
        -- TODO: different lookup for farm
        e_link_to_fifo : entity work.link_to_fifo
        generic map (
            g_LOOPUP_NAME        => g_LOOPUP_NAME,
            is_FARM              => true,
            SKIP_DOUBLE_SUB      => false,
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_rx            => i_rx(i),
            i_linkid        => work.mudaq.link_36_to_std(i),

            o_q             => o_data(i),
            i_ren           => i_ren(i),
            o_rdempty       => o_empty(i),

            o_counter(0)    => o_counter(0+i*5),
            o_counter(1)    => o_counter(1+i*5),
            o_counter(2)    => o_counter(2+i*5),
            o_counter(3)    => o_counter(3+i*5),
            o_counter(4)    => o_counter(4+i*5),

            i_reset_n       => i_reset_n,
            i_clk           => i_clk--,
        );

-- Spring IntRun22 we dont align since we have on farm for pixel and one for scifi
--        --! align links and send data to the next farm
--        e_aligne_link : entity work.farm_aligne_link
--        generic map (
--            g_NLINKS_SWB_TOTL    => g_NLINKS_SWB_TOTL,
--            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
--        )
--        port map (
--            i_rx        => rx_q(i),
--            i_sop       => sop,
--            i_sop_cur   => sop(i),
--            i_eop       => eop(i),
--            o_skip      => skip(i),
--            i_skip      => skip,
--            
--            i_empty     => rx_rdempty,
--            i_empty_cur => rx_rdempty(i),
--            o_ren       => rx_ren(i),
--            
--            o_tx        => o_tx(i),
--            o_tx_k      => o_tx_k(i),
--
--            --! error counters 
--            --! 0: fifo sync_almost_full
--            --! 1: fifo sync_wrfull
--            --! 2: # of next farm event
--            --! 3: cnt events
--            o_counter(0)    => o_counter(0+i*4+g_NLINKS_SWB_TOTL*5),
--            o_counter(1)    => o_counter(1+i*4+g_NLINKS_SWB_TOTL*5),
--            o_counter(2)    => o_counter(2+i*4+g_NLINKS_SWB_TOTL*5),
--            o_counter(3)    => o_counter(3+i*4+g_NLINKS_SWB_TOTL*5),
--            o_data          => data(i),
--            o_empty         => o_empty(i),
--            i_ren           => i_ren(i),
--            
--            o_error         => o_error(i),
--
--            i_reset_n       => i_reset_n,
--            i_clk           => i_clk--,
--        );
--        
--        -- map outputs
--        o_sop(i)      <= '1' when data(i)(34 downto 32) = "010" else '0';
--        o_shop(i)     <= '1' when data(i)(34 downto 32) = "111" else '0';
--        o_eop(i)      <= '1' when data(i)(34 downto 32) = "001" else '0';
--        o_hit(i)      <= '1' when data(i)(34 downto 32) = "000" else '0';
--        o_t0(i)       <= '1' when data(i)(34 downto 32) = "100" else '0';
--        o_t1(i)       <= '1' when data(i)(34 downto 32) = "101" else '0';
--        o_data(i)     <= data(i)(31 downto 0);
    
    END GENERATE;

end architecture;
