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


entity farm_link_merger is
    generic(
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

        i_link_data : in std_logic_vector(NLINKS_TOTL * 32 - 1 downto 0);
        i_link_datak : in std_logic_vector(NLINKS_TOTL * 4 - 1 downto 0);
        i_link_valid : in integer;
        i_link_mask_n : in std_logic_vector(NLINKS_TOTL - 1 downto 0);

        o_stream_rdata : out std_logic_vector(W - 1 downto 0); -- "11" = shop, "10" = eop, "01" = sop, "00" = data
        o_stream_rempty : out std_logic;
        i_stream_rack : in std_logic--;

    );
    end entity;

    architecture arch of farm_link_merger is
             
        signal reset_data, reset_mem : std_logic;
        
        signal link_data, link_dataq : data_array(NLINKS_TOTL - 1 downto 0);
        signal link_empty, link_wren, link_full, link_afull, link_wrfull, sop, eop, shop, link_ren : std_logic_vector(NLINKS_TOTL - 1 downto 0);
        signal link_usedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH * NLINKS_TOTL - 1 downto 0);
        signal sync_fifo_empty : std_logic_vector(NLINKS_TOTL - 1 downto 0);
        signal sync_fifo_i_wrreq : std_logic_vector(NLINKS_TOTL - 1 downto 0);
        type sync_fifo_t is array (NLINKS_TOTL - 1 downto 0) of std_logic_vector(35 downto 0);
        signal sync_fifo_q : sync_fifo_t;
        signal sync_fifo_data : sync_fifo_t;
        
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
            sync_fifo_data(i) <= (others => '0');
            sync_fifo_i_wrreq(i) <= '0';
        elsif ( rising_edge(i_dataclk) ) then
            sync_fifo_data(i) <= i_link_data(31 + i * 32 downto i * 32) & i_link_datak(3 + i * 4 downto i * 4);
            if ( i_link_data(31 + i * 32 downto i * 32) = x"000000BC" and i_link_datak(3 + i * 4 downto i * 4) = "0001" ) then
                sync_fifo_i_wrreq(i) <= '0';
            else
                sync_fifo_i_wrreq(i) <= '1';
            end if;
        end if;
    end process;
    
    e_sync_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 6,
        DATA_WIDTH  => 36--,
    )
    port map (
        data        => sync_fifo_data(i),
        wrreq       => sync_fifo_i_wrreq(i),
        rdreq       => not sync_fifo_empty(i),
        wrclk       => i_dataclk,
        rdclk       => i_memclk,
        q           => sync_fifo_q(i),
        rdempty     => sync_fifo_empty(i),
        aclr        => '0'--,
    );
    
    e_link_to_fifo : entity work.link_to_fifo
    generic map(
        W => 32--,
    )
    port map(
        i_link_data         => sync_fifo_q(i)(35 downto 4),
        i_link_datak        => sync_fifo_q(i)(3 downto 0),
        i_fifo_almost_full  => link_afull(i),
        i_sync_fifo_empty   => sync_fifo_empty(i),
        o_fifo_data         => link_data(i)(35 downto 0),
        o_fifo_wr           => link_wren(i),
        o_cnt_skip_data     => open,
        i_reset_n           => i_reset_mem_n,
        i_clk               => i_memclk--,
    );
    
    -- sop
    link_data(i)(36) <= '1' when ( link_data(i)(3 downto 0) = "0001" and link_data(i)(11 downto 4) = x"BC" ) else '0';
    -- eop
    link_data(i)(37) <= '1' when ( link_data(i)(3 downto 0) = "0001" and link_data(i)(11 downto 4) = x"9C" ) else '0';

    e_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH  => 38,
        SHOWAHEAD   => "ON"--,
    )
    port map (
        data        => link_data(i),
        wrreq       => link_wren(i),
        rdreq       => link_ren(i),
        wrclk       => i_dataclk,
        rdclk       => i_memclk,
        q           => link_dataq(i),
        rdempty     => link_empty(i),
        rdusedw     => open,
        wrfull      => open,
        wrusedw     => link_usedw(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1 downto i * LINK_FIFO_ADDR_WIDTH),
        aclr        => reset_data--,
    );
    
    process(i_dataclk, i_reset_data_n)
    begin
        if(i_reset_data_n = '0') then
            link_afull(i)       <= '0';
        elsif(rising_edge(i_dataclk)) then
            if(link_usedw(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1) = '1') then
                link_afull(i)   <= '1';
            else 
                link_afull(i)   <= '0';
            end if;
        end if;
    end process;
    
    sop(i) <= link_dataq(i)(36);
    shop(i) <= '1' when link_dataq(i)(37 downto 36) = "00" and link_dataq(I)(31 downto 26) = "111111" else '0';
    eop(i) <= link_dataq(i)(37);

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
        i_rsop                  => sop,
        i_reop                  => eop,
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
