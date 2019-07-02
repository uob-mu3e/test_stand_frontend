library ieee;
use ieee.std_logic_1164.all;

package ddr3_components is

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

	
	




end package ddr3_components;