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
    g_LOOPUP_NAME        : string := "intRun2021";
    g_NLINKS_SWB_TOTL    : positive :=  3;
    N_PIXEL              : positive :=  2;
    N_SCIFI              : positive :=  1;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in  work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    i_rx_k          : in  work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    
    o_tx            : out  work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    o_tx_k          : out  work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    
    -- data out for farm path
    o_data          : out work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    o_empty         : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    i_ren           : in  std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_shop          : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_sop           : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_eop           : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_hit           : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_t0            : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_t1            : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_error         : out std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    
    --! status counters 
    --! (g_NLINKS_DATA*5)-1 downto 0 -> link to fifo counters
    --! (g_NLINKS_DATA*4)+(g_NLINKS_DATA*5)-1 downto (g_NLINKS_DATA*5) -> link align counters
    o_counter       : out work.util.slv32_array_t((g_NLINKS_SWB_TOTL*4)+(g_NLINKS_SWB_TOTL*5)-1 downto 0);
    
    i_clk_250_link      : in std_logic;
    i_reset_n_250_link  : in std_logic;

    i_clk_250       : in std_logic;
    i_reset_n_250   : in std_logic--;
);
end entity;

architecture arch of farm_link_to_fifo is

    signal rx_q, data : work.util.slv35_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_ren, rx_mask_n, rx_rdempty : std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0) := (others => '0');
    signal sop, eop, skip : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0);
    
    signal rx_pixel : work.util.slv34_array_t(N_PIXEL - 1 downto 0);
    signal rx_scifi : work.util.slv34_array_t(N_SCIFI - 1 downto 0);

begin

    --! sync link data from link to pcie clk
    gen_link_to_fifo : FOR i in 0 to g_NLINKS_SWB_TOTL - 1 GENERATE
    
        -- TODO: different lookup for farm
        e_link_to_fifo_32 : entity work.link_to_fifo_32
        generic map (
            g_LOOPUP_NAME        => g_LOOPUP_NAME,
            is_FARM              => true,
            SKIP_DOUBLE_SUB      => false,
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_rx            => i_rx(i),
            i_rx_k          => i_rx_k(i),
            i_linkid        => work.mudaq.link_36_to_std(i),
            o_tx            => o_tx(i),
            o_tx_k          => o_tx_k(i),

            o_q             => rx_q(i),
            i_ren           => rx_ren(i),
            o_rdempty       => rx_rdempty(i),

            o_counter(0)    => o_counter(0+i*5),
            o_counter(1)    => o_counter(1+i*5),
            o_counter(2)    => o_counter(2+i*5),
            o_counter(3)    => o_counter(3+i*5),
            o_counter(4)    => o_counter(4+i*5),

            i_reset_n_156   => i_reset_n_156,
            i_clk_156       => i_clk_156,

            i_reset_n_250   => reset_250_n,
            i_clk_250       => i_clk_250--,
        );

        -- map outputs
        sop(i)      <= '1' when rx_q(i)(34 downto 32) = "010" else '0';
        shop(i)     <= '1' when rx_q(i)(34 downto 32) = "111" else '0';
        eop(i)      <= '1' when rx_q(i)(34 downto 32) = "001" else '0';
        hit(i)      <= '1' when rx_q(i)(34 downto 32) = "000" else '0';
        t0(i)       <= '1' when rx_q(i)(34 downto 32) = "100" else '0';
        t1(i)       <= '1' when rx_q(i)(34 downto 32) = "101" else '0';
 
    END GENERATE gen_link_to_fifo;
    
    
    --! align links and send data to the next farm
    gen_align_links : FOR i in 0 to g_NLINKS_SWB_TOTL - 1 GENERATE
        
        e_aligne_link : entity work.farm_aligne_link
        generic map (
            g_NLINKS_SWB_TOTL    => g_NLINKS_SWB_TOTL,
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_rx    => rx_q(i),
            i_sop   => sop,
            i_sop_cur => sop(i),
            i_eop   => eop(i),
            o_skip  => skip(i),
            i_skip  => skip,
            
            i_empty => rx_rdempty,
            i_empty_cur => rx_rdempty(i),
            o_ren   => rx_ren(i),
            
            o_tx    => o_tx(i),
            o_tx_k  => o_tx_k(i),

            --! error counters 
            --! 0: fifo sync_almost_full
            --! 1: fifo sync_wrfull
            --! 2: # of next farm event
            --! 3: cnt events
            o_counter(0)    => o_counter(0+i*4+g_NLINKS_SWB_TOTL*5),
            o_counter(1)    => o_counter(1+i*4+g_NLINKS_SWB_TOTL*5),
            o_counter(2)    => o_counter(2+i*4+g_NLINKS_SWB_TOTL*5),
            o_counter(3)    => o_counter(3+i*4+g_NLINKS_SWB_TOTL*5),
            o_data          => data(i),
            o_empty         => o_empty(i),
            i_ren           => i_ren(i),
            
            o_error         => o_error(i),

            i_reset_n_250   => i_reset_n_250,
            i_clk_250       => i_clk_250--,
        );
        
        -- map outputs
        o_sop(i)      <= '1' when data(i)(34 downto 32) = "010" else '0';
        o_shop(i)     <= '1' when data(i)(34 downto 32) = "111" else '0';
        o_eop(i)      <= '1' when data(i)(34 downto 32) = "001" else '0';
        o_hit(i)      <= '1' when data(i)(34 downto 32) = "000" else '0';
        o_t0(i)       <= '1' when data(i)(34 downto 32) = "100" else '0';
        o_t1(i)       <= '1' when data(i)(34 downto 32) = "101" else '0';
        o_data(i)     <= data(i)(31 downto 0);
    
    END GENERATE;

end architecture;
