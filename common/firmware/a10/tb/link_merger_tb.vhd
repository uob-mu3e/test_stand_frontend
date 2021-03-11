library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.dataflow_components.all;



entity link_merger_tb is 
end entity link_merger_tb;


architecture TB of link_merger_tb is

    signal reset_n		: std_logic;
    signal reset		: std_logic;
    signal enable_pix : std_logic := '0';

    -- Input from merging (first board) or links (subsequent boards)
    signal dataclk, stream_we		: 		 std_logic;
    
    -- links and datageneration
    -- use 38 links but mask last 2 since only 34 are in use
    constant NLINKS_TOTL : integer := 64;
    constant NLINKS_DATA : integer := 36;
    constant LINK_FIFO_ADDR_WIDTH : integer := 10;
    
    signal link_data : std_logic_vector(NLINKS_TOTL * 32 - 1 downto 0);
    signal link_mask_n : std_logic_vector(NLINKS_TOTL - 1 downto 0);
    signal link_datak : std_logic_vector(NLINKS_TOTL * 4 - 1 downto 0);
   
    signal slow_down          : std_logic_vector(31 downto 0);
    signal data_tiles_generated : std_logic_vector(31 downto 0);
    signal data_scifi_generated : std_logic_vector(31 downto 0);
    signal data_pix_generated : std_logic_vector(31 downto 0);
    signal datak_tiles_generated : std_logic_vector(3 downto 0);
    signal datak_scifi_generated : std_logic_vector(3 downto 0);
    signal datak_pix_generated : std_logic_vector(3 downto 0);
    signal we_counter : std_logic_vector(31 downto 0);
    signal stream_rempty : std_logic;

    -- clk period
    constant dataclk_period : time := 4 ns;

begin

    -- data generation and ts counter_ddr3
    slow_down <= x"00000002";

    e_data_gen_mupix : entity work.data_generator_a10
    generic map (
        go_to_trailer => 4,
        go_to_sh => 3--,
    )
    port map(
        clk => dataclk,
        reset => reset,
        enable_pix => enable_pix,
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
    generic map (
        go_to_trailer => 4,
        go_to_sh => 3--,
    )
    port map(
        clk => dataclk,
        reset => reset,
        enable_pix => enable_pix,
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
    generic map (
        go_to_trailer => 4,
        go_to_sh => 3--,
    )
    port map(
        clk => dataclk,
        reset => reset,
        enable_pix => enable_pix,
        i_dma_half_full => '0',
        random_seed => (others => '1'),
        start_global_time => (others => '0'),
        data_pix_generated => data_tiles_generated,
        datak_pix_generated => datak_tiles_generated,
        data_pix_ready => open,
        slow_down => slow_down,
        state_out => open--,
    );
    
    link_data <= x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        &
                 x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        & x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        &
                 x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        & x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        &
                 x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        & x"00000000"        & x"00000000"          & x"00000000"          & x"00000000"        &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & 
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_pix_generated & data_scifi_generated ;

    link_datak <= x"0"                & x"0"                  & x"0"                  & x"0"              & 
                  x"0"                & x"0"                  &  x"0"                 & x"0"              & x"0"                  & x"0"                  & x"0"                  & x"0"                & 
                  x"0"                & x"0"                  &  x"0"                 & x"0"              & x"0"                  & x"0"                  & x"0"                  & x"0"                & 
                  x"0"                & x"0"                  &  x"0"                 & x"0"              & x"0"                  & x"0"                  & x"0"                  & x"0"                &
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated &
                  datak_pix_generated & datak_scifi_generated & datak_pix_generated & datak_scifi_generated ;
                 
--     link_mask_n <= "1101110111011101110111011101110111";
    --link_mask_n <= "11111111111111111111111111111111";
        --link_mask_n <= "1111111111111111111111111111111111";
--     link_mask_n <= "1111111111111111";
--     link_mask_n <= "11011101";
--     link_mask_n <= "11111111";
    link_mask_n <= x"000000000000000F";

    e_link_merger : entity work.link_merger
    generic map(
        NLINKS_TOTL => NLINKS_TOTL,
        TREE_DEPTH_w  => 5,
        TREE_DEPTH_r  => 5,
        LINK_FIFO_ADDR_WIDTH => 8--,
    )
    port map(
        i_reset_data_n => reset_n,
        i_reset_mem_n => reset_n,
        i_dataclk => dataclk,
        i_memclk => dataclk,
	
		i_link_data => link_data,
		i_link_datak => link_datak,
		i_link_valid => 1,
		i_link_mask_n => link_mask_n,
		
		o_stream_rdata => open,
		o_stream_rempty => stream_rempty,
		i_stream_rack => not stream_rempty--,
    );

	--dataclk
	process begin
		dataclk <= '0';
		wait for dataclk_period/2;
		dataclk <= '1';
		wait for dataclk_period/2;
	end process;
	
    reset <= not reset_n;
    
    inita : process
    begin
	   reset_n	 <= '0';
	   wait for 8 ns;
	   reset_n	 <= '1';
	   wait for 20 ns;
	   enable_pix    <= '1';
	   wait;
    end process inita;
    
end TB;


