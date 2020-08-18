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
		i_link_mask_n : in  std_logic_vector (NLINKS_TOTL - 1 downto 0)--;
		
	);
	end entity link_merger;
	
	architecture RTL of link_merger is
             
        signal reset_data, reset_mem : std_logic;
        
        signal link_data, link_dataq : data_array(NLINKS_TOTL - 1 downto 0);
        signal link_empty, link_wren, link_full, link_afull, link_wrfull, sop, eop, link_ren : std_logic_vector(NLINKS_TOTL - 1 downto 0);
        signal link_usedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH * NLINKS_TOTL - 1 downto 0);
        
        signal stream_in_rempty : std_logic_vector(NLINKS_TOTL-1 downto 0);
        signal stream_wdata, stream_rdata : std_logic_vector(35 downto 0);
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
    eop(i) <= link_dataq(i)(37); 

    END GENERATE buffer_link_fifos;
    
    stream_in_rempty <= link_empty;
    
    e_time_merger : entity work.time_merger
        generic map (
        W => 38,
        N => NLINKS_TOTL--,
    )
    port map (
        i_rdata                 => link_dataq,
        i_rsop                  => sop,
        i_reop                  => eop,
        i_rempty                => stream_in_rempty,
        i_mask_n                => i_link_mask_n,
        o_rack                  => link_ren,

        o_wdata(35 downto 0)    => stream_wdata,
        i_wfull                 => stream_wfull,
        o_we                    => stream_we,

        i_reset_n               => i_reset_mem_n,
        i_clk                   => i_memclk--,
    );
    
    e_stream_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 36,
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

    end architecture RTL;
