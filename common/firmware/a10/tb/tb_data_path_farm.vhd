library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;

entity tb_data_path_farm is 
end entity tb_data_path_farm;


architecture TB of tb_data_path_farm is

    signal reset_n		: std_logic;
    signal reset		: std_logic;

    -- Input from merging (first board) or links (subsequent boards)
    signal dataclk		: 		 std_logic;
    signal data_en		:		 std_logic;
    signal data_in		:		 std_logic_vector(511 downto 0);
    signal ts_in		:		 std_logic_vector(31 downto 0);

    -- Input from PCIe demanding events
    signal pcieclk		:		std_logic;
    signal ts_req_A		:		std_logic_vector(31 downto 0);
    signal req_en_A		:		std_logic;
    signal ts_req_B		:		std_logic_vector(31 downto 0);
    signal req_en_B		:		std_logic;
    signal tsblock_done :		std_logic_vector(15 downto 0);

    -- Output to DMA
    signal dma_data_out	    :	std_logic_vector(255 downto 0);
    signal dma_data_en		:	std_logic;
    signal dma_eoe			:   std_logic;

    -- Interface to memory bank A
    signal A_mem_clk		: std_logic;
    signal A_mem_ready		: std_logic;
    signal A_mem_calibrated	: std_logic;
    signal A_mem_addr		: std_logic_vector(25 downto 0);
    signal A_mem_data		: std_logic_vector(511 downto 0);
    signal A_mem_write		: std_logic;
    signal A_mem_read		: std_logic;
    signal A_mem_q			: std_logic_vector(511 downto 0);
    signal A_mem_q_valid	: std_logic;

    -- Interface to memory bank B
    signal B_mem_clk		: std_logic;
    signal B_mem_ready		: std_logic;
    signal B_mem_calibrated	: std_logic;
    signal B_mem_addr		: std_logic_vector(25 downto 0);
    signal B_mem_data		: std_logic_vector(511 downto 0);
    signal B_mem_write		: std_logic;
    signal B_mem_read		: std_logic;
    signal B_mem_q			: std_logic_vector(511 downto 0);
    signal B_mem_q_valid	: std_logic;
    
    -- links and datageneration
    constant NLINKS     : positive := 8;
    constant NLINKS_TOTL: positive := 16;
    constant LINK_FIFO_ADDR_WIDTH : integer := 10;
    
    signal link_data        : std_logic_vector(NLINKS * 32 - 1 downto 0);
    signal link_datak       : std_logic_vector(NLINKS * 4 - 1 downto 0);
    signal counter_ddr3     : std_logic_vector(31 downto 0);
    
    signal w_pixel, r_pixel, w_scifi, r_scifi : std_logic_vector(NLINKS * 38 -1 downto 0);
    signal w_pixel_en, r_pixel_en, full_pixel, empty_pixel : std_logic;
    signal w_scifi_en, r_scifi_en, full_scifi, empty_scifi : std_logic;
    signal FEB_num : work.util.slv6_array_t(5 downto 0);
    
    signal rx : work.util.slv32_array_t(NLINKS_TOTL-1 downto 0);
    signal rx_k : work.util.slv4_array_t(NLINKS_TOTL-1 downto 0);
    
    signal link_data_pixel, link_data_scifi : std_logic_vector(NLINKS * 32 - 1  downto 0);
    signal link_datak_pixel, link_datak_scifi : std_logic_vector(NLINKS * 4 - 1  downto 0);
    
    signal pixel_data, scifi_data : std_logic_vector(257 downto 0);
    signal pixel_empty, pixel_ren, scifi_empty, scifi_ren : std_logic;
    signal data_wen, endofevent, ddr_ready : std_logic;
    signal event_ts : std_logic_vector(47 downto 0);
    signal ts_req_num : std_logic_vector(31 downto 0);
    
    -- clk period
    constant dataclk_period : time := 4 ns;
    constant pcieclk_period : time := 4 ns;
    constant A_mem_clk_period : time := 3.76 ns;
    constant B_mem_clk_period : time := 3.76 ns;


    signal toggle : std_logic_vector(1 downto 0);
    signal startinput : std_logic;
    signal ts_in_next			:		 std_logic_vector(31 downto 0);

    signal A_mem_read_del1: std_logic;
    signal A_mem_read_del2: std_logic;
    signal A_mem_read_del3: std_logic;
    signal A_mem_read_del4: std_logic;

    signal A_mem_addr_del1		: std_logic_vector(25 downto 0);
    signal A_mem_addr_del2		: std_logic_vector(25 downto 0);
    signal A_mem_addr_del3		: std_logic_vector(25 downto 0);
    signal A_mem_addr_del4		: std_logic_vector(25 downto 0);

    signal B_mem_read_del1: std_logic;
    signal B_mem_read_del2: std_logic;
    signal B_mem_read_del3: std_logic;
    signal B_mem_read_del4: std_logic;

    signal B_mem_addr_del1		: std_logic_vector(25 downto 0);
    signal B_mem_addr_del2		: std_logic_vector(25 downto 0);
    signal B_mem_addr_del3		: std_logic_vector(25 downto 0);
    signal B_mem_addr_del4		: std_logic_vector(25 downto 0);	


begin

    -- gen pixel data
    e_data_gen_pixel : entity work.data_generator_merged_data
    port map(
        i_clk       => dataclk,
        i_reset_n   => reset_n,
        i_en        => not full_pixel,
        i_sd        => x"00000002",
        o_data      => w_pixel,
        o_data_we   => w_pixel_en,
        o_state     => open--,
    );
    
    e_merger_fifo_pixel : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_pixel,
        wrreq           => w_pixel_en,
        rdreq           => r_pixel_en,
        clock           => dataclk,
        q               => r_pixel,
        full            => full_pixel,
        empty           => empty_pixel,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => reset--,
    );
    
    e_swb_data_merger_pixel : entity work.swb_data_merger
    generic map (
        NLINKS      => NLINKS,
        DATA_TYPE   => x"01"--,
    )
    port map (
        i_reset_n   => reset_n,
        i_clk       => dataclk,
        
        i_data      => r_pixel,
        i_empty     => empty_pixel,
        
        o_ren       => r_pixel_en,
        o_wen       => open,
        o_data      => link_data_pixel,
        o_datak     => link_datak_pixel--,
    );
    

    -- gen scifi data
    e_data_gen_scifi : entity work.data_generator_merged_data
    port map(
        i_clk       => dataclk,
        i_reset_n   => reset_n,
        i_en        => not full_scifi,
        i_sd        => x"00000002",
        o_data      => w_scifi,
        o_data_we   => w_scifi_en,
        o_state     => open--,
    );
    
    e_merger_fifo_scifi : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_scifi,
        wrreq           => w_scifi_en,
        rdreq           => r_scifi_en,
        clock           => dataclk,
        q               => r_scifi,
        full            => full_scifi,
        empty           => empty_scifi,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => reset--,
    );
    
    e_swb_data_merger_scifi : entity work.swb_data_merger
    generic map (
        NLINKS      => NLINKS,
        DATA_TYPE   => x"02"--,
    )
    port map (
        i_reset_n   => reset_n,
        i_clk       => dataclk,
        
        i_data      => r_scifi,
        i_empty     => empty_scifi,
        
        o_ren       => r_scifi_en,
        o_wen       => open,
        o_data      => link_data_scifi,
        o_datak     => link_datak_scifi--,
    );
    
    -- map links
    gen_mapping : FOR I in NLINKS-1 to 0 GENERATE
        rx(I) <= link_data_pixel(I*32 + 31 downto I*32);
        rx_k(I) <= link_datak_pixel(I*4 + 3 downto I*4);
        rx(I+NLINKS) <= link_data_scifi(I*32 + 31 downto I*32);
        rx_k(I+NLINKS) <= link_datak_scifi(I*4 + 3 downto I*4);
    END GENERATE;
    
    e_data_demerge_pixel : entity work.farm_link_to_fifo
    generic map (
        g_NLINKS_SWB_TOTL   => NLINKS_TOTL,
        N_PIXEL             => NLINKS,
        N_SCIFI             => NLINKS--,
    )
    port map (
        i_rx                => rx,
        i_rx_k              => rx_k,

        -- pixel data
        o_pixel             => pixel_data,
        o_empty_pixel       => pixel_empty, 
        i_ren_pixel         => pixel_ren,
        o_error_pixel       => open,
        
        -- scifi data
        o_scifi             => scifi_data,
        o_empty_scifi       => scifi_empty,
        i_ren_scifi         => scifi_ren,
        o_error_scifi       => open,
    
        --! error counters 
        --! 0: fifo f_almost_full
        --! 1: fifo f_wrfull
        --! 2: # of skip event
        --! 3: # of events
        o_counter           => open, -- out work.util.slv32_array_t(4 * g_NLINKS_SWB_TOTL - 1 downto 0);
        
        i_clk_250_link      => dataclk,
        i_reset_n_250_link  => reset_n,
        
        i_clk_250           => dataclk,  -- should be DDR clk
        i_reset_n_250       => reset_n--,
    );
    
    
    
    
    e_farm_midas_event_builder : entity work.farm_midas_event_builder
    generic map (
        g_NLINKS_SWB_TOTL => 16,
        N_PIXEL           => 8,
        N_SCIFI           => 8,
        RAM_ADDR_W        => 12,
        RAM_ADDR_R        => 18--,
    )
    port map (
        i_pixel         => pixel_data,
        i_empty_pixel   => pixel_empty,
        o_ren_pixel     => pixel_ren,
    
        i_scifi         => scifi_data,
        i_empty_scifi   => scifi_empty,
        o_ren_scifi     => pixel_ren,

        -- DDR
        o_data          => data_in,
        o_wen           => data_wen,
        o_endofevent    => endofevent,
        o_event_ts      => event_ts,
        i_ddr_ready     => ddr_ready,
    
        -- Link data
        o_pixel         => open,
        o_wen_pixel     => open,
    
        o_scifi         => open,
        o_wen_scifi     => open,

        o_counters      => open,

        i_reset_n_250   => reset_n,
        i_clk_250       => dataclk--,
    );

    process(dataclk, reset_n)
    begin
        if( reset_n <= '0' ) then
            counter_ddr3    <= (others => '0');
        elsif ( dataclk'event and dataclk = '1' ) then
            counter_ddr3 <= counter_ddr3 + '1';
        end if;
    end process;

    e_farm_data_path : entity work.farm_data_path 
    port map(
        reset_n         => reset_n,
        reset_n_ddr3    => reset_n,

        -- Input from merging (first board) or links (subsequent boards)
        dataclk         => dataclk,
        data_in         => data_in,
        data_en         => data_wen, 
        ts_in           => event_ts(35 downto 4), -- 3:0 -> hit, 9:0 -> sub header
        o_ddr_ready     => ddr_ready,

        -- Input from PCIe demanding events
        pcieclk         => pcieclk,
        ts_req_A        => ts_req_A,
        req_en_A        => req_en_A,
        ts_req_B        => ts_req_B,
        req_en_B        => req_en_B,
        tsblock_done    => tsblock_done,

        -- Output to DMA
        dma_data_out    => dma_data_out,
        dma_data_en     => dma_data_en,
        dma_eoe         => dma_eoe,
        i_dmamemhalffull=> '0',
        i_num_req_events=> ts_req_num,
        o_dma_done      => open,
        i_dma_wen       => '1',

        -- Interface to memory bank A
        A_mem_clk       => A_mem_clk,
        A_mem_ready     => A_mem_ready,
        A_mem_calibrated=> A_mem_calibrated,
        A_mem_addr      => A_mem_addr,
        A_mem_data      => A_mem_data,
        A_mem_write     => A_mem_write,
        A_mem_read      => A_mem_read,
        A_mem_q         => A_mem_q,
        A_mem_q_valid   => A_mem_q_valid,

        -- Interface to memory bank B
        B_mem_clk       => B_mem_clk,
        B_mem_ready     => B_mem_ready,
        B_mem_calibrated=> B_mem_calibrated,
        B_mem_addr		=> B_mem_addr,
        B_mem_data		=> B_mem_data,
        B_mem_write		=> B_mem_write,
        B_mem_read		=> B_mem_read,
        B_mem_q			=> B_mem_q,
        B_mem_q_valid	=> B_mem_q_valid
	);
	
    e_ddr3_a : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => 9,
        ADDR_WIDTH_B    => 9,
        DATA_WIDTH_A    => 512,
        DATA_WIDTH_B    => 512,
        DEVICE          => "Arria 10"--,
    )
    port map (
        address_a       => A_mem_addr(8 downto 0),
        address_b       => A_mem_addr(8 downto 0),
        clock_a         => A_mem_clk,
        clock_b         => A_mem_clk,
        data_a          => A_mem_data,
        data_b          => (others => '0'),
        wren_a          => A_mem_write,
        wren_b          => '0',
        q_a             => open,
        q_b             => A_mem_q--,
    );
    
    e_ddr3_b : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => 9,
        ADDR_WIDTH_B    => 9,
        DATA_WIDTH_A    => 512,
        DATA_WIDTH_B    => 512,
        DEVICE          => "Arria 10"--,
    )
    port map (
        address_a       => B_mem_addr(8 downto 0),
        address_b       => B_mem_addr(8 downto 0),
        clock_a         => B_mem_clk,
        clock_b         => B_mem_clk,
        data_a          => B_mem_data,
        data_b          => (others => '0'),
        wren_a          => B_mem_write,
        wren_b          => '0',
        q_a             => open,
        q_b             => B_mem_q--,
    );
	
	--dataclk
	process begin
		dataclk <= '0';
		wait for dataclk_period/2;
		dataclk <= '1';
		wait for dataclk_period/2;
	end process;
	
	--pcieclk
	process begin
		pcieclk <= '0';
		wait for pcieclk_period/2;
		pcieclk <= '1';
		wait for pcieclk_period/2;
	end process;
	
	--A_mem_clk
	process begin
		A_mem_clk <= '0';
		wait for A_mem_clk_period/2;
		A_mem_clk <= '1';
		wait for A_mem_clk_period/2;
	end process;
	
	-- Reset_n
	process begin
		reset_n <= '0';
		startinput <= '0';
		wait for 20 ns;
		reset_n <= '1';
		wait for 200 ns;
		startinput <= '1';
		wait;
	end process;

    reset <= not reset_n;

	-- Memready
	process begin
		A_mem_ready <= '0';
		B_mem_ready <= '0';
		wait for A_mem_clk_period * 25;
		A_mem_ready <= '1';
		B_mem_ready <= '1';
		wait for A_mem_clk_period * 300;
		A_mem_ready <= '0';
		B_mem_ready <= '0';
		wait for A_mem_clk_period;
		A_mem_ready <= '1';
		B_mem_ready <= '1';
		wait for A_mem_clk_period * 250;
		A_mem_ready <= '0';
		B_mem_ready <= '0';
		wait for A_mem_clk_period;
		A_mem_ready <= '1';
		B_mem_ready <= '1';
		wait for A_mem_clk_period * 600;
	end process;

	A_mem_calibrated <= '1';
	B_mem_calibrated <= '1';

	-- Request generation
	process begin
	req_en_A <= '0';	
	wait for pcieclk_period;-- * 26500;
	req_en_A <= '1';
	ts_req_num <= x"00000008";
	ts_req_A <= x"04030201";--"00010000"; 	
	wait for pcieclk_period;
	req_en_A <= '1';
	ts_req_A <= x"0B0A0906";--x"00030002"; 	
	wait for pcieclk_period;
	req_en_A <= '0';
	wait for pcieclk_period;
	req_en_A <= '0';
	tsblock_done	<= (others => '0');
	end process;


	-- Memory A simulation
	process(A_mem_clk, reset_n)
	begin
	if(reset_n <= '0') then
		A_mem_q_valid 	<= '0';
		A_mem_read_del1 <= '0';
		A_mem_read_del2 <= '0';
		A_mem_read_del3 <= '0';
		A_mem_read_del4 <= '0';
	elsif(A_mem_clk'event and A_mem_clk = '1') then
		A_mem_read_del1 <= A_mem_read;
		A_mem_read_del2 <= A_mem_read_del1;
		A_mem_read_del3	<= A_mem_read_del2;
		A_mem_read_del4	<= A_mem_read_del3;
		A_mem_q_valid   <= A_mem_read_del4;

		A_mem_addr_del1 <= A_mem_addr;
		A_mem_addr_del2 <= A_mem_addr_del1;
		A_mem_addr_del3	<= A_mem_addr_del2;
		A_mem_addr_del4	<= A_mem_addr_del3;
-- 		A_mem_q		<= (others => '0');
-- 		A_mem_q(25 downto 0)  <= A_mem_addr_del4;
	end if;
	end process;


	-- Memory B simulation
	process(B_mem_clk, reset_n)
	begin
	if(reset_n <= '0') then
		B_mem_q_valid 	<= '0';
		B_mem_read_del1 <= '0';
		B_mem_read_del2 <= '0';
		B_mem_read_del3 <= '0';
		B_mem_read_del4 <= '0';
	elsif(B_mem_clk'event and B_mem_clk = '1') then
		B_mem_read_del1 <= B_mem_read;
		B_mem_read_del2 <= B_mem_read_del1;
		B_mem_read_del3	<= B_mem_read_del2;
		B_mem_read_del4	<= B_mem_read_del3;
		B_mem_q_valid   <= B_mem_read_del4;

		B_mem_addr_del1 <= B_mem_addr;
		B_mem_addr_del2 <= B_mem_addr_del1;
		B_mem_addr_del3	<= B_mem_addr_del2;
		B_mem_addr_del4	<= B_mem_addr_del3;
-- 		B_mem_q		<= (others => '0');
-- 		B_mem_q(25 downto 0)  <= B_mem_addr_del4;
	end if;
	end process;
end TB;


