-----------------------------------------------------------------------------
-- Handling of the DDR3 (eventually DDR4) buffer for the farm pCs
--
-- Niklaus Berger, JGU Mainz
-- niberger@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.ddr3_components.all;





entity ddr3_block is 
	port (
			clk					  : in 	 std_logic;   	-- clock for the FPGA fabric side
			reset_n				  : in 	 std_logic;		-- global reset

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
	end entity ddr3_block;
	
	architecture RTL of ddr3_block is
	
	signal A_cal_success:	std_logic;
	signal A_cal_fail:		std_logic;
	
	signal A_clk:				std_logic;
	signal A_reset:			std_logic;
	
	signal A_ready:			std_logic;
	signal A_read:				std_logic;
	signal A_write:			std_logic;
	signal A_address:			std_logic_vector(25 downto 0);
	signal A_readdata:		std_logic_vector(511 downto 0);
	signal A_writedata:		std_logic_vector(511 downto 0);
	signal A_burstcount:		std_logic_vector(6 downto 0);
	signal A_readdatavalid: std_logic;
	
	
	signal B_cal_success:	std_logic;
	signal B_cal_fail:		std_logic;
	
	signal B_clk:				std_logic;
	signal B_reset:			std_logic;
	
	signal B_ready:			std_logic;
	signal B_read:				std_logic;
	signal B_write:			std_logic;
	signal B_address:			std_logic_vector(25 downto 0);
	signal B_readdata:		std_logic_vector(511 downto 0);
	signal B_writedata:		std_logic_vector(511 downto 0);
	signal B_burstcount:		std_logic_vector(6 downto 0);
	signal B_readdatavalid: std_logic;
	
	begin

	ddr3_A:ddr3_if
		port map(
			amm_ready_0         => A_ready,
			amm_read_0          => A_read,
			amm_write_0         => A_write,
			amm_address_0       => A_address,
			amm_readdata_0      => A_readdata,
			amm_writedata_0     => A_writedata,
			amm_burstcount_0    => A_burstcount,
			amm_byteenable_0    =>  (others => '1'), 
			amm_readdatavalid_0 => A_readdatavalid,  
			emif_usr_clk        => A_clk,
			emif_usr_reset_n    => A_reset,
			global_reset_n      => reset_n,
			mem_ck              => A_mem_ck,
			mem_ck_n            => A_mem_ck_n,
			mem_a               => A_mem_a,
			mem_ba              => A_mem_ba,
			mem_cke             => A_mem_cke,
			mem_cs_n            => A_mem_cs_n,
			mem_odt             => A_mem_odt,
			mem_reset_n         => A_mem_reset_n,
			mem_we_n            => A_mem_we_n,
			mem_ras_n           => A_mem_ras_n,
			mem_cas_n           => A_mem_cas_n,
			mem_dqs             => A_mem_dqs,
			mem_dqs_n           => A_mem_dqs_n,
			mem_dq              => A_mem_dq,
			mem_dm              => A_mem_dm,
			oct_rzqin           => A_oct_rzqin,
			pll_ref_clk         => A_pll_ref_clk,
			local_cal_success   => A_cal_success,
			local_cal_fail      => A_cal_fail
		);

		ddr3_B:ddr3_if
		port map(
			amm_ready_0         => B_ready,
			amm_read_0          => B_read,
			amm_write_0         => B_write,
			amm_address_0       => B_address,
			amm_readdata_0      => B_readdata,
			amm_writedata_0     => B_writedata,
			amm_burstcount_0    => B_burstcount,
			amm_byteenable_0    =>  (others => '1'), 
			amm_readdatavalid_0 => B_readdatavalid,  
			emif_usr_clk        => B_clk,
			emif_usr_reset_n    => B_reset,
			global_reset_n      => reset_n,
			mem_ck              => B_mem_ck,
			mem_ck_n            => B_mem_ck_n,
			mem_a               => B_mem_a,
			mem_ba              => B_mem_ba,
			mem_cke             => B_mem_cke,
			mem_cs_n            => B_mem_cs_n,
			mem_odt             => B_mem_odt,
			mem_reset_n         => B_mem_reset_n,
			mem_we_n            => B_mem_we_n,
			mem_ras_n           => B_mem_ras_n,
			mem_cas_n           => B_mem_cas_n,
			mem_dqs             => B_mem_dqs,
			mem_dqs_n           => B_mem_dqs_n,
			mem_dq              => B_mem_dq,
			mem_dm              => B_mem_dm,
			oct_rzqin           => B_oct_rzqin,
			pll_ref_clk         => B_pll_ref_clk,
			local_cal_success   => B_cal_success,
			local_cal_fail      => B_cal_fail
		);

	
	
	
	end architecture RTL;
	
	
	