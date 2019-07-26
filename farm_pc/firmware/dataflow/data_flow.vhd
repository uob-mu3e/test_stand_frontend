-----------------------------------------------------------------------------
-- Handling of the data flow for the farm PCs
--
-- Niklaus Berger, JGU Mainz
-- niberger@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
--use work.ddr3_components.all;
--use work.pcie_components.all;




entity data_flow is 
	port (
			reset_n		: 		in std_logic;
			
			-- Input from merging (first board) or links (subsequent boards)
			dataclk		: 		in std_logic;
			data_en		:		in std_logic;
			data_in		:		in std_logic_vector(255 downto 0);
			ts_in			:		in	std_logic_vector(31 downto 0);
			
			-- Input from PCIe demanding events
			pcieclk		:		in std_logic;
			ts_req		:		in std_logic_vector(31 downto 0);
			req_en		:		in std_logic;
			
			-- Output to DMA
			dma_data_out	:	out std_logic_vector(255 downto 0);
			dma_data_en		:	out std_logic;
			dma_eoe			:  out std_logic;
			
			-- Output to links
			link_data_out	:	out std_logic_vector(255 downto 0);
			link_ts_out		:	out std_logic_vector(31 downto 0);
			link_data_en	:	out std_logic;
			
			-- Interface to memory bank A
			A_mem_ck              : out   std_logic_vector(0 downto 0);                      -- mem_ck
			A_mem_ck_n            : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
			A_mem_a               : out   std_logic_vector(15 downto 0);                     -- mem_a
			A_mem_ba              : out   std_logic_vector(2 downto 0);                      -- mem_ba
			A_mem_cke             : out   std_logic_vector(0 downto 0);                      -- mem_cke
			A_mem_cs_n            : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
			A_mem_odt             : out   std_logic_vector(0 downto 0);                      -- mem_odt
			A_mem_reset_n         : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
			A_mem_we_n            : out   std_logic_vector(0 downto 0);                      -- mem_we_n
			A_mem_ras_n           : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
			A_mem_cas_n           : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
			A_mem_dqs             : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
			A_mem_dqs_n           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
			A_mem_dq              : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
			A_mem_dm              : out   std_logic_vector(7 downto 0);                      -- mem_dm
			A_oct_rzqin           : in    std_logic                      := 'X';             -- oct_rzqin
			A_pll_ref_clk         : in    std_logic                      := 'X';             -- clk

			-- Interface to memory bank B
			B_mem_ck              : out   std_logic_vector(0 downto 0);                      -- mem_ck
			B_mem_ck_n            : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
			B_mem_a               : out   std_logic_vector(15 downto 0);                     -- mem_a
			B_mem_ba              : out   std_logic_vector(2 downto 0);                      -- mem_ba
			B_mem_cke             : out   std_logic_vector(0 downto 0);                      -- mem_cke
			B_mem_cs_n            : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
			B_mem_odt             : out   std_logic_vector(0 downto 0);                      -- mem_odt
			B_mem_reset_n         : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
			B_mem_we_n            : out   std_logic_vector(0 downto 0);                      -- mem_we_n
			B_mem_ras_n           : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
			B_mem_cas_n           : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
			B_mem_dqs             : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
			B_mem_dqs_n           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
			B_mem_dq              : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
			B_mem_dm              : out   std_logic_vector(7 downto 0);                      -- mem_dm
			B_oct_rzqin           : in    std_logic                      := 'X';             -- oct_rzqin
			B_pll_ref_clk         : in    std_logic                      := 'X'             -- clk
	
	);
	end entity data_flow;
	
	architecture RTL of data_flow is
	
	begin
	
	end architecture RTL;
	