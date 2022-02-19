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
    --! (g_NLINKS_DATA*5)-1 downto 0 -> link to fifo counters
    o_counter       : out work.util.slv32_array_t((g_NLINKS_DATA*5)-1 downto 0);
    
    i_clk_250_link      : in std_logic;
    i_reset_n_250_link  : in std_logic;

    i_clk_250       : in std_logic;
    i_reset_n_250   : in std_logic--;
);
end entity;

architecture arch of farm_link_to_fifo is

    signal rx_q : work.util.slv35_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_q_s : work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_ren, rx_mask_n, rx_rdempty : std_logic_vector(g_NLINKS_DATA-1 downto 0) := (others => '0');
    signal sop, eop, shop, hit, t0, t1 : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0);
    
    signal rx_pixel : work.util.slv34_array_t(N_PIXEL - 1 downto 0);
    signal rx_scifi : work.util.slv34_array_t(N_SCIFI - 1 downto 0);

begin

    --! sync link data from link to pcie clk
    gen_link_to_fifo : FOR i in 0 to g_NLINKS_SWB_TOTL - 1 GENERATE
    
        -- TODO: If its halffull than send data to the next farm
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
        rx_q_s(i)   <= rx_q(i)(31 downto 0);
 
    END GENERATE gen_link_to_fifo;
    
    gen_map_pixel : FOR i in 0 to N_PIXEL - 1 GENERATE
        rx_pixel(i) <= rx_q_s(i);
    END GENERATE;
    
    gen_map_scifi : FOR i in 0 to N_SCIFI - 1 GENERATE
        rx_scifi(i) <= rx_q_s(i+N_PIXEL);
    END GENERATE;
    
    -- TODO: add time merging tree here
    -- TODO:
    -- TODO:
    
    
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

        i_reset_n_250   => i_reset_n_250,
        i_clk_250       => i_clk_250--,
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

        i_reset_n_250   => i_reset_n_250,
        i_clk_250       => i_clk_250--,
    );

end architecture;
