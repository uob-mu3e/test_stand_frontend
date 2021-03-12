-------------------------------------------------------
--! @link_to_fifo_32.vhd
--! @brief the link_to_fifo_32 sorts out the data from the 
--! link and provides it as a fifo (32b package)
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity link_to_fifo_32 is
generic (
    LINK_FIFO_ADDR_WIDTH : integer := 10--;
);
port (
    i_rx            : in std_logic_vector(31 downto 0);
    i_rx_k          : in std_logic_vector(3 downto 0);
    
    o_q             : out std_logic_vector(37 downto 0);
    i_ren           : in std_logic;
    o_rdempty       : out std_logic;

    --! error counters 
    --! 0: fifo almost_full
    --! 1: fifo wrfull
    --! 2: # of skip event
    o_counter     : out work.util.slv32_array_t(2 downto 0);

    i_reset_n_156   : in std_logic;
    i_clk_156       : in std_logic;

    i_reset_n_250   : in std_logic;
    i_clk_250       : in std_logic--;
);
end entity;

architecture arch of link_to_fifo_32 is
   
    type link_to_fifo_type is (idle, write_data, skip_data);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_data : std_logic_vector(31 downto 0);

    signal rx_156_data, rx_156_q : std_logic_vector(35 downto 0);
    signal rx_250_data : std_logic_vector(37 downto 0);
    signal rx_156_wen, rx_250_wen, sync_rdempty, almost_full, wrfull : std_logic;
    signal wrusedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH - 1 downto 0);

begin

    e_cnt_link_fifo_almost_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(0), i_ena => almost_full, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    e_cnt_dc_link_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(1), i_ena => wrfull, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    --! sync data from data clk to dma clk
    --! write only if not idle
    process(i_clk_156, i_reset_n_156)
    begin
        if ( i_reset_n_156 = '0' ) then
            rx_156_data  <= (others => '0');
            rx_156_wen    <= '0';
        elsif ( rising_edge(i_clk_156) ) then
            rx_156_data <= i_rx & i_rx_k;
            if ( i_rx = x"000000BC" and i_rx_k = "0001" ) then
                rx_156_wen <= '0';
            else
                rx_156_wen <= '1';
            end if;
        end if;
    end process;
        
    e_sync_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 6,
        DATA_WIDTH  => 36,
        DEVICE      => "Arria 10"--,
    )
    port map (
        data        => rx_156_data,
        wrreq       => rx_156_wen,
        rdreq       => not sync_rdempty,
        wrclk       => i_clk_156,
        rdclk       => i_clk_250,
        q           => rx_156_q,
        rdempty     => sync_rdempty,
        aclr        => '0'--,
    );
        
    e_link_to_fifo : entity work.link_to_fifo
    generic map(
        W => 32--,
    )
    port map(
        i_link_data         => rx_156_q(35 downto 4),
        i_link_datak        => rx_156_q(3 downto 0),
        i_fifo_almost_full  => almost_full,
        i_sync_fifo_empty   => sync_rdempty,
        o_fifo_data         => rx_250_data(35 downto 0),
        o_fifo_wr           => rx_250_wen,
        o_cnt_skip_data     => o_counter(2),
        i_reset_n           => i_reset_n_250,
        i_clk               => i_clk_250--,
    );
        
    -- sop
    rx_250_data(36) <= '1' when ( rx_250_data(3 downto 0) = "0001" and rx_250_data(11 downto 4) = x"BC" ) else '0';
    -- eop
    rx_250_data(37) <= '1' when ( rx_250_data(3 downto 0) = "0001" and rx_250_data(11 downto 4) = x"9C" ) else '0';
    
    e_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH  => 38,
        DEVICE      => "Arria 10"--,
    )
    port map (
        data        => rx_250_data,
        wrreq       => rx_250_wen,
        rdreq       => i_ren,
        wrclk       => i_clk_250,
        rdclk       => i_clk_250,
        q           => o_q,
        rdempty     => o_rdempty,
        rdusedw     => open,
        wrfull      => wrfull,
        wrusedw     => wrusedw,
        aclr        => not i_reset_n_250--,
    );

    process(i_clk_250, i_reset_n_250)
    begin
        if(i_reset_n_250 = '0') then
            almost_full       <= '0';
        elsif(rising_edge(i_clk_250)) then
            if(wrusedw(LINK_FIFO_ADDR_WIDTH - 1) = '1') then
                almost_full <= '1';
            else 
                almost_full <= '0';
            end if;
        end if;
    end process;

end architecture;
