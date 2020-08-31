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
use work.dataflow_components.all;


entity link_merger is
    generic(
        NLINKS_TOTL : integer := 3;
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

        o_stream_rdata : out std_logic_vector(67 downto 0); -- "11" = shop, "10" = eop, "01" = sop, "00" = data
        o_stream_rempty : out std_logic;
        i_stream_rack : in std_logic--;

    );
    end entity link_merger;

    architecture RTL of link_merger is
             
        signal reset_data, reset_mem : std_logic;
        
        signal link_data, link_dataq : data_array(NLINKS_TOTL - 1 downto 0);
        signal link_empty, link_wren, link_full, link_afull, link_wrfull, sop, eop, shop, link_ren : std_logic_vector(NLINKS_TOTL - 1 downto 0);
        signal link_usedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH * NLINKS_TOTL - 1 downto 0);
        
        signal stream_wdata, stream_rdata : std_logic_vector(67 downto 0);
        signal we_counter : std_logic_vector(64 downto 0);
        signal stream_rempty, stream_rack, stream_wfull, stream_we : std_logic;
        
	begin
	
    reset_data <= not i_reset_data_n;
    reset_mem <= not i_reset_mem_n;
    
    buffer_link_fifos:
    FOR i in 0 to NLINKS_TOTL - 1 GENERATE

    e_link_to_fifo : entity work.link_to_fifo
    generic map(
        W => 32--,
    )
    port map(
        i_link_data         => i_link_data(31 + i * 32 downto i * 32),
        i_link_datak        => i_link_datak(3 + i * 4 downto i * 4),
        i_fifo_almost_full  => link_afull(i),
        o_fifo_data         => link_data(i)(35 downto 0),
        o_fifo_wr           => link_wren(i),
        o_cnt_skip_data     => open,
        i_reset_n           => i_reset_data_n,
        i_clk               => i_dataclk--,
    );
    
    -- sop
    link_data(i)(36) <= '1' when ( link_data(i)(3 downto 0) = "0001" and link_data(i)(11 downto 4) = x"BC" ) else '0';
    -- eop
    link_data(i)(37) <= '1' when ( link_data(i)(3 downto 0) = "0001" and link_data(i)(11 downto 4) = x"9C" ) else '0';

    e_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH  => 38,
        SHOWAHEAD   => "ON",
        DEVICE      => "Arria 10"--,
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
        W => 68,
        N => NLINKS_TOTL--,
    )
    port map (
        i_rdata                 => link_dataq,
        i_rsop                  => sop,
        i_reop                  => eop,
        i_rshop                 => shop,
        i_rempty                => link_empty,
        i_link                  => i_link_valid,
        i_mask_n                => i_link_mask_n,
        o_rack                  => link_ren,

        o_wdata                 => stream_wdata,
        o_wsop                  => open,
        o_weop                  => open,
        i_wfull                 => stream_wfull,
        o_we                    => stream_we,

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
    
    e_stream_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 10,
        DATA_WIDTH => 68,
        DEVICE => "Arria 10"--,
    )
    port map (
        q               => stream_rdata,
        empty           => stream_rempty,
        rdreq           => stream_rack,
        data            => stream_wdata,
        full            => stream_wfull,
        wrreq           => stream_we,
        sclr            => reset_mem,
        clock           => i_memclk--,
    );
        
    o_stream_rdata <= stream_rdata;
    o_stream_rempty <= stream_rempty;
    stream_rack <= i_stream_rack;

    end architecture RTL;
