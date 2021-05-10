library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


entity tb_data_path_farm is 
end entity tb_data_path_farm;


architecture TB of tb_data_path_farm is

    signal reset_n		: std_logic;
    signal reset		: std_logic;

    -- Input from merging (first board) or links (subsequent boards)
    signal dataclk		: 		 std_logic;
    signal data_en		:		 std_logic_vector(2 downto 0);
    signal data_in		:		 work.util.slv256_array_t(2 downto 0);
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

    -- Output to links -- with dataclk
    signal link_data_out	:	std_logic_vector(255 downto 0);
    signal link_ts_out		:	std_logic_vector(31 downto 0);
    signal link_data_en	    :	std_logic;

    -- Interface to memory bank A
    signal A_mem_clk		: std_logic;
    signal A_mem_ready		: std_logic;
    signal A_mem_calibrated	: std_logic;
    signal A_mem_addr		: std_logic_vector(25 downto 0);
    signal A_mem_data		: std_logic_vector(255 downto 0);
    signal A_mem_write		: std_logic;
    signal A_mem_read		: std_logic;
    signal A_mem_q			: std_logic_vector(255 downto 0);
    signal A_mem_q_valid	: std_logic;

    -- Interface to memory bank B
    signal B_mem_clk		: std_logic;
    signal B_mem_ready		: std_logic;
    signal B_mem_calibrated	: std_logic;
    signal B_mem_addr		: std_logic_vector(25 downto 0);
    signal B_mem_data		: std_logic_vector(255 downto 0);
    signal B_mem_write		: std_logic;
    signal B_mem_read		: std_logic;
    signal B_mem_q			: std_logic_vector(255 downto 0);
    signal B_mem_q_valid	: std_logic;
    
    -- links and datageneration
    constant NLINKS     : positive := 8;
    constant LINK_FIFO_ADDR_WIDTH : integer := 10;
    
    signal link_data        : std_logic_vector(NLINKS * 32 - 1 downto 0);
    signal link_datak       : std_logic_vector(NLINKS * 4 - 1 downto 0);
    signal counter_ddr3     : std_logic_vector(31 downto 0);
    
    signal w_fifo_data, r_fifo_data : std_logic_vector(NLINKS * 38 -1 downto 0);
    signal w_fifo_en, r_fifo_en, fifo_full, fifo_empty : std_logic;
    signal FEB_num : work.util.slv6_array_t(5 downto 0);
    
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

    e_data_gen : entity work.data_generator_merged_data
    port map(
        i_clk       => dataclk,
        i_reset_n   => reset_n,
        i_en        => not fifo_full,
        i_sd        => x"00000002",
        o_data      => w_fifo_data,
        o_data_we   => w_fifo_en,
        o_state     => open--,
    );
    
    e_merger_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_fifo_data,
        wrreq           => w_fifo_en,
        rdreq           => r_fifo_en,
        clock           => dataclk,
        q               => r_fifo_data,
        full            => fifo_full,
        empty           => fifo_empty,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => reset--,
    );
    
    -- data merger swb
    e_swb_data_merger : entity work.swb_data_merger
    generic map (
        NLINKS  => NLINKS,
        DATA_TYPE      => x"01"--,
    )
    port map (
        i_reset_n   => reset_n,
        i_clk       => dataclk,
        
        i_data      => r_fifo_data,
        i_empty     => fifo_empty,
        
        o_ren       => r_fifo_en,
        o_wen       => open,
        o_data      => link_data,
        o_datak     => link_datak--,
    );

    FEB_num(0) <= link_data(37 downto 32);
    FEB_num(1) <= link_data(75 downto 70);
    FEB_num(2) <= link_data(113 downto 108);
    FEB_num(3) <= link_data(151 downto 146);
    FEB_num(4) <= link_data(189 downto 184);
    FEB_num(5) <= link_data(227 downto 222);

    data_in(0) <= link_data;
    data_in(1) <= link_data;
    data_in(2) <= link_data;
    data_en(0) <= '0' when link_data(7 downto 0) = x"BC" and link_datak(3 downto 0) = "0001" else '1';
    data_en(1) <= '0' when link_data(7 downto 0) = x"BC" and link_datak(3 downto 0) = "0001" else '1';
    data_en(2) <= '0' when link_data(7 downto 0) = x"BC" and link_datak(3 downto 0) = "0001" else '1';

    process(dataclk, reset_n)
    begin
        if( reset_n <= '0' ) then
            counter_ddr3    <= (others => '0');
        elsif ( dataclk'event and dataclk = '1' ) then
            counter_ddr3 <= counter_ddr3 + '1';
        end if;
    end process;

    dut : entity work.farm_data_path 
    port map(
        reset_n         => reset_n,
        reset_n_ddr3    => reset_n,

        -- Input from merging (first board) or links (subsequent boards)
        dataclk         => dataclk,
        data_en         => data_en(0),
        data_in         => data_in(0),
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


