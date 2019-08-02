library ieee;
use ieee.std_logic_1164.all;

use work.pcie_components.all;

package ddr3_components is

	component ddr3_block is 
	port (
			reset_n	: in std_logic;
			
			-- Control and status registers
			ddr3control			: in reg32;
			ddr3status			: out reg32;
			
			-- A interface
			A_ddr3clk				: out std_logic;
			A_ddr3calibrated		: out std_logic;
			A_ddr3ready				: out std_logic;
			A_ddr3addr				: in std_logic_vector(25 downto 0);
			A_ddr3datain			: in std_logic_vector(511 downto 0);
			A_ddr3dataout			: out std_logic_vector(511 downto 0);
			A_ddr3_write			: in std_logic;
			A_ddr3_read				: in std_logic;
			A_ddr3_read_valid		: out std_logic;
			
			-- B interface
			B_ddr3clk				: out std_logic;
			B_ddr3calibrated		: out std_logic;
			B_ddr3ready				: out std_logic;
			B_ddr3addr				: in std_logic_vector(25 downto 0);
			B_ddr3datain			: in std_logic_vector(511 downto 0);
			B_ddr3dataout			: out std_logic_vector(511 downto 0);
			B_ddr3_write			: in std_logic;
			B_ddr3_read				: in std_logic;
			B_ddr3_read_valid		: out std_logic;
		
			-- Error counters
			errout					: out reg32;	
		
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
	end component;



		component ddr3_if is
		port (
			amm_ready_0         : out   std_logic;                                         -- waitrequest_n
			amm_read_0          : in    std_logic                      := 'X';             -- read
			amm_write_0         : in    std_logic                      := 'X';             -- write
			amm_address_0       : in    std_logic_vector(25 downto 0)  := (others => 'X'); -- address
			amm_readdata_0      : out   std_logic_vector(511 downto 0);                    -- readdata
			amm_writedata_0     : in    std_logic_vector(511 downto 0) := (others => 'X'); -- writedata
			amm_burstcount_0    : in    std_logic_vector(6 downto 0)   := (others => 'X'); -- burstcount
			amm_byteenable_0    : in    std_logic_vector(63 downto 0)  := (others => 'X'); -- byteenable
			amm_readdatavalid_0 : out   std_logic;                                         -- readdatavalid
			emif_usr_clk        : out   std_logic;                                         -- clk
			emif_usr_reset_n    : out   std_logic;                                         -- reset_n
			global_reset_n      : in    std_logic                      := 'X';             -- reset_n
			mem_ck              : out   std_logic_vector(0 downto 0);                      -- mem_ck
			mem_ck_n            : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
			mem_a               : out   std_logic_vector(15 downto 0);                     -- mem_a
			mem_ba              : out   std_logic_vector(2 downto 0);                      -- mem_ba
			mem_cke             : out   std_logic_vector(0 downto 0);                      -- mem_cke
			mem_cs_n            : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
			mem_odt             : out   std_logic_vector(0 downto 0);                      -- mem_odt
			mem_reset_n         : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
			mem_we_n            : out   std_logic_vector(0 downto 0);                      -- mem_we_n
			mem_ras_n           : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
			mem_cas_n           : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
			mem_dqs             : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
			mem_dqs_n           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
			mem_dq              : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
			mem_dm              : out   std_logic_vector(7 downto 0);                      -- mem_dm
			oct_rzqin           : in    std_logic                      := 'X';             -- oct_rzqin
			pll_ref_clk         : in    std_logic                      := 'X';             -- clk
			local_cal_success   : out   std_logic;                                         -- local_cal_success
			local_cal_fail      : out   std_logic                                          -- local_cal_fail
		);
	end component ddr3_if;

	
component ddr3_memory_controller is 
	port (
		reset_n	: in std_logic;
		
		-- Control and status registers
		ddr3control		: in std_logic_vector(15 downto 0);
		ddr3status		: out std_logic_vector(15 downto 0);
		
		ddr3clk			: out std_logic;
		ddr3_calibrated: out std_logic;
		ddr3_ready		: out std_logic;
		
		ddr3addr			: in std_logic_vector(25 downto 0);
		ddr3datain		: in std_logic_vector(511 downto 0);
		ddr3dataout		: out std_logic_vector(511 downto 0);
		ddr3_write		: in std_logic;
		ddr3_read		: in std_logic;
		ddr3_read_valid: out std_logic;
		
		-- Error counters
		errout			: out reg32;

		-- IF to DDR3 
		M_cal_success	:	in std_logic;
		M_cal_fail		:	in	std_logic;
		M_clk				:	in	std_logic;
		M_reset_n		:	in	std_logic;
		M_ready			:	in	std_logic;
		M_read			:	out std_logic;
		M_write			:	out std_logic;
		M_address		:  out std_logic_vector(25 downto 0);
		M_readdata		:	in	std_logic_vector(511 downto 0);
		M_writedata		:	out std_logic_vector(511 downto 0);
		M_burstcount	:	out std_logic_vector(6 downto 0);
		M_readdatavalid:  in std_logic
	);
end component ddr3_memory_controller;




end package ddr3_components;