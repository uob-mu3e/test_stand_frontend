library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.dataflow_components.all;



entity data_flow_tb is 
end entity data_flow_tb;


architecture TB of data_flow_tb is
	
			signal reset_n		: std_logic;
			
			-- Input from merging (first board) or links (subsequent boards)
			signal dataclk		: 		 std_logic;
			signal data_en		:		 std_logic;
			signal data_in		:		 std_logic_vector(255 downto 0);
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
			
			
			constant dataclk_period : time := 4 ns;
			constant pcieclk_period : time := 4 ns;
			constant A_mem_clk_period : time := 3.76 ns;
			constant B_mem_clk_period : time := 3.76 ns;
			
			
			signal toggle : std_logic_vector(1 downto 0);
			signal startinput : std_logic;
			signal ts_in_next			:		 std_logic_vector(31 downto 0);
begin

dut:data_flow 
	port map(
			reset_n		=> reset_n,
			
			-- Input from merging (first board) or links (subsequent boards)
			dataclk		=> dataclk,
			data_en		=> data_en,
			data_in		=> data_in,
			ts_in			=> ts_in,
			
			-- Input from PCIe demanding events
			pcieclk		=> pcieclk,
			ts_req_A		=> ts_req_A,
			req_en_A		=> req_en_A,
			ts_req_B		=> ts_req_B,
			req_en_B		=> req_en_B,
			tsblock_done=> tsblock_done,
			
			-- Output to DMA
			dma_data_out	=> dma_data_out,
			dma_data_en		=> dma_data_en,
			dma_eoe			=> dma_eoe,
			
			-- Output to links -- with dataclk
			link_data_out	=> link_data_out,
			link_ts_out		=> link_ts_out,
			link_data_en	=> link_data_en,
			
			-- Interface to memory bank A
			A_mem_clk		=> A_mem_clk,
			A_mem_ready		=> A_mem_ready,
			A_mem_calibrated	=> A_mem_calibrated,
			A_mem_addr		=> A_mem_addr,
			A_mem_data		=> A_mem_data,
			A_mem_write		=> A_mem_write,
			A_mem_read		=> A_mem_read,
			A_mem_q			=> A_mem_q,
			A_mem_q_valid	=> A_mem_q_valid,

			-- Interface to memory bank B
			B_mem_clk		=> B_mem_clk,
			B_mem_ready		=> B_mem_ready,
			B_mem_calibrated	=> B_mem_calibrated,
			B_mem_addr		=> B_mem_addr,
			B_mem_data		=> B_mem_data,
			B_mem_write		=> B_mem_write,
			B_mem_read		=> B_mem_read,
			B_mem_q			=> B_mem_q,
			B_mem_q_valid	=> B_mem_q_valid
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

	-- Memready
	process begin
		A_mem_ready <= '0';
		B_mem_ready <= '0';
		wait for 100 ns;
		A_mem_ready <= '1';
		B_mem_ready <= '1';
		wait for 1200 ns;
		A_mem_ready <= '0';
		B_mem_ready <= '0';
		wait for 4 ns;
		A_mem_ready <= '1';
		B_mem_ready <= '1';
		wait for 800 ns;
		A_mem_ready <= '0';
		B_mem_ready <= '0';
		wait for 4 ns;
		A_mem_ready <= '1';
		B_mem_ready <= '1';
		wait for 2000 ns;
	end process;



	A_mem_calibrated <= '1';
	B_mem_calibrated <= '1';

	-- Data generation
	process(dataclk, reset_n)
	begin
	if(reset_n <= '0') then
		data_en <= '0';
		ts_in_next   <= (others => '0');
		toggle <= "00";
	elsif(dataclk'event and dataclk = '1') then
		ts_in <= ts_in_next;
		if(startinput = '1') then
			data_en <= '1';
			toggle <= toggle + '1';
			if(toggle = "00") then
				ts_in_next <= ts_in_next + '1';
			end if;
			if(toggle = "01") then
				ts_in_next <= ts_in_next;
			end if;
			if(toggle = "10") then
				ts_in_next <= ts_in_next + '1';
			end if;
			if(toggle = "01") then
				ts_in_next <= ts_in_next + "1000";
			end if;

			data_in <= toggle & toggle & x"A" & x"BC1234" &
					  ts_in_next & ts_in_next & ts_in_next & ts_in_next & ts_in_next & ts_in_next & ts_in_next;
		end if;
	end if;
	end process;
	
end TB;


