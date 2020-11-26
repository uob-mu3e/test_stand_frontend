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
    constant NLINKS_TOTL : integer := 32;
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

    -- clk period
    constant dataclk_period : time := 4 ns;

begin

    -- data generation and ts counter_ddr3
    slow_down <= x"00000002";

    e_data_gen_mupix : entity work.data_generator_a10
    generic map (
      go_to_trailer => "0000111111"
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
      go_to_trailer => "0000111111"
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
      go_to_trailer => "0000111111"
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
    
    link_data <= data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & 
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated when NLINKS_TOTL = 34 else 
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & 
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated when NLINKS_TOTL = 32 else 
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated &
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated when NLINKS_TOTL = 16 else 
                 data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated & data_pix_generated & data_scifi_generated & data_tiles_generated & data_pix_generated;
    link_datak <= datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated &
                  datak_pix_generated & datak_scifi_generated when NLINKS_TOTL = 34 else
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated when NLINKS_TOTL = 32 else
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated when NLINKS_TOTL = 16 else 
                  datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated & datak_pix_generated & datak_scifi_generated & datak_tiles_generated & datak_pix_generated;
                  
--     link_mask_n <= "1101110111011101110111011101110111";
    link_mask_n <= "11111111111111111111111111111111";
        --link_mask_n <= "1111111111111111111111111111111111";
--     link_mask_n <= "1111111111111111";
--     link_mask_n <= "11011101";
--     link_mask_n <= "11111111";
    
    e_link_merger : entity work.link_merger
    generic map(
        NLINKS_TOTL => NLINKS_TOTL,
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
		o_stream_rempty => open,
		i_stream_rack => '1'--,
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


