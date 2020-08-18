library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.dataflow_components.all;



entity data_flow_tb is 
end entity data_flow_tb;


architecture TB of data_flow_tb is

    signal reset_n		: std_logic;
    signal reset		: std_logic;

    -- Input from merging (first board) or links (subsequent boards)
    signal dataclk		: 		 std_logic;
    signal data_en		:		 std_logic;
    signal data_in		:		 std_logic_vector(255 downto 0) := (others => '0');
    signal ts_in			:		 std_logic_vector(31 downto 0);

    -- Input from PCIe demanding events
    signal pcieclk		:		std_logic;
    signal ts_req_A		:		std_logic_vector(31 downto 0);
    signal req_en_A		:		std_logic;
    signal ts_req_B		:		std_logic_vector(31 downto 0);
    signal req_en_B		:		std_logic;
    signal tsblock_done:		std_logic_vector(15 downto 0);

    -- Output to DMA
    signal dma_data_out	:	std_logic_vector(255 downto 0);
    signal dma_data_en		:	std_logic;
    signal dma_eoe			:  std_logic;

    -- Output to links -- with dataclk
    signal link_data_out	:	std_logic_vector(255 downto 0);
    signal link_ts_out		:	std_logic_vector(31 downto 0);
    signal link_data_en	:	std_logic;

    -- Interface to memory bank A
    signal A_mem_clk		: std_logic;
    signal A_mem_ready		: std_logic;
    signal A_mem_calibrated		: std_logic;
    signal A_mem_addr		: std_logic_vector(25 downto 0);
    signal A_mem_data		: std_logic_vector(255 downto 0);
    signal A_mem_write		: std_logic;
    signal A_mem_read		: std_logic;
    signal A_mem_q			: std_logic_vector(255 downto 0);
    signal A_mem_q_valid	: std_logic;

    -- Interface to memory bank B
    signal B_mem_clk		: std_logic;
    signal B_mem_ready		: std_logic;
    signal B_mem_calibrated		: std_logic;
    signal B_mem_addr		: std_logic_vector(25 downto 0);
    signal B_mem_data		: std_logic_vector(255 downto 0);
    signal B_mem_write		: std_logic;
    signal B_mem_read		: std_logic;
    signal B_mem_q			: std_logic_vector(255 downto 0);
    signal B_mem_q_valid	: std_logic;
    
    -- links and datageneration
    constant NLINKS_TOTL : integer := 3;
    constant LINK_FIFO_ADDR_WIDTH : integer := 10;
    
    signal link_data : std_logic_vector(NLINKS_TOTL * 32 - 1 downto 0);
    signal link_datak : std_logic_vector(NLINKS_TOTL * 4 - 1 downto 0);
    signal counter_ddr3          : std_logic_vector(31 downto 0);
    
    signal slow_down          : std_logic_vector(31 downto 0);
    signal data_tiles_generated : std_logic_vector(31 downto 0);
    signal data_scifi_generated : std_logic_vector(31 downto 0);
    signal data_pix_generated : std_logic_vector(31 downto 0);
    signal datak_tiles_generated : std_logic_vector(3 downto 0);
    signal datak_scifi_generated : std_logic_vector(3 downto 0);
    signal datak_pix_generated : std_logic_vector(3 downto 0);

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

    -- data generation and ts counter_ddr3
    slow_down <= x"00000002";
    
    process(A_mem_clk, reset_n)
    begin
    if(reset_n <= '0') then
        counter_ddr3 	<= (others => '0');
    elsif(A_mem_clk'event and A_mem_clk = '1') then
        counter_ddr3 <= counter_ddr3 + '1';
    end if;
    end process;

    e_data_gen_mupix : entity work.data_generator_a10
    port map(
        clk => dataclk,
        reset => reset,
        enable_pix => '1',
        i_dma_half_full => '0',
        random_seed => (others => '1'),
        start_global_time => (others => '0'),
        data_pix_generated => data_pix_generated,
        datak_pix_generated => datak_pix_generated,
        data_pix_ready => open,
        slow_down => slow_down,
        state_out => open--,
    );
    
    e_data_gen_scifi : entity work.data_generator_a10
    port map(
        clk => dataclk,
        reset => reset,
        enable_pix => '1',
        i_dma_half_full => '0',
        random_seed => (others => '1'),
        start_global_time => (others => '0'),
        data_pix_generated => data_scifi_generated,
        datak_pix_generated => datak_scifi_generated,
        data_pix_ready => open,
        slow_down => slow_down,
        state_out => open--,
    );
    
    e_data_gen_tiles : entity work.data_generator_a10
    port map(
        clk => dataclk,
        reset => reset,
        enable_pix => '1',
        i_dma_half_full => '0',
        random_seed => (others => '1'),
        start_global_time => (others => '0'),
        data_pix_generated => data_tiles_generated,
        datak_pix_generated => datak_tiles_generated,
        data_pix_ready => open,
        slow_down => slow_down,
        state_out => open--,
    );
    
    link_data <= data_pix_generated & data_scifi_generated & data_tiles_generated;
    link_datak <= datak_pix_generated & datak_scifi_generated & datak_tiles_generated;
    
    dut : entity work.data_flow 
    port map(
        reset_n         => reset_n,
        reset_n_ddr3    => reset_n,

        -- Input from merging (first board) or links (subsequent boards)
        dataclk         => dataclk,
        data_en         => link_fifo_ren,
        data_in         => data_in,
        ts_in           => counter_ddr3,

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

        -- Output to links -- with dataclk
        link_data_out   => link_data_out,
        link_ts_out     => link_ts_out,
        link_data_en    => link_data_en,

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
        DATA_WIDTH_A    => 256,
        DATA_WIDTH_B    => 256,
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
        DATA_WIDTH_A    => 256,
        DATA_WIDTH_B    => 256,
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
	ts_req_A <= x"00000001";--"00010000"; 	
	wait for pcieclk_period;
	req_en_A <= '1';
	ts_req_A <= x"00000002";--x"00030002"; 	
	wait for pcieclk_period;
	req_en_A <= '1';
	ts_req_A <= x"00000003"; 	
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


