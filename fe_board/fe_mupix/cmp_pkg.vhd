library ieee;
use ieee.std_logic_1164.all;
package cmp is
--Copyright (C) 2018  Intel Corporation. All rights reserved.
--Your use of Intel Corporation's design tools, logic functions 
--and other software and tools, and its AMPP partner logic 
--functions, and any output files from any of the foregoing 
--(including device programming or simulation files), and any 
--associated documentation or information are expressly subject 
--to the terms and conditions of the Intel Program License 
--Subscription Agreement, the Intel Quartus Prime License Agreement,
--the Intel FPGA IP License Agreement, or other applicable license
--agreement, including, without limitation, that your use is for
--the sole purpose of programming logic devices manufactured by
--Intel and sold by Intel or its authorized distributors.  Please
--refer to the applicable agreement for further details.


component ip_altgx_reconfig
	PORT
	(
		reconfig_clk		: IN STD_LOGIC ;
		reconfig_fromgxb		: IN STD_LOGIC_VECTOR (16 DOWNTO 0);
		reconfig_reset		: IN STD_LOGIC ;
		busy		: OUT STD_LOGIC ;
		error		: OUT STD_LOGIC ;
		reconfig_togxb		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0)
	);
end component;
--Copyright (C) 2018  Intel Corporation. All rights reserved.
--Your use of Intel Corporation's design tools, logic functions 
--and other software and tools, and its AMPP partner logic 
--functions, and any output files from any of the foregoing 
--(including device programming or simulation files), and any 
--associated documentation or information are expressly subject 
--to the terms and conditions of the Intel Program License 
--Subscription Agreement, the Intel Quartus Prime License Agreement,
--the Intel FPGA IP License Agreement, or other applicable license
--agreement, including, without limitation, that your use is for
--the sole purpose of programming logic devices manufactured by
--Intel and sold by Intel or its authorized distributors.  Please
--refer to the applicable agreement for further details.


component ip_altgx
    generic (
        effective_data_rate : positive := 6250;
        input_clock_frequency : string := "125.0";
        input_clock_period : positive := 1000000 / 125; -- us
        m_divider : positive := 25--;
    );
	PORT
	(
		cal_blk_clk		: IN STD_LOGIC ;
		pll_inclk		: IN STD_LOGIC ;
		pll_powerdown		: IN STD_LOGIC_VECTOR (0 DOWNTO 0);
		reconfig_clk		: IN STD_LOGIC ;
		reconfig_togxb		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_analogreset		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_coreclk		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_datain		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_digitalreset		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_enapatternalign		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_seriallpbken		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		tx_coreclk		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		tx_ctrlenable		: IN STD_LOGIC_VECTOR (15 DOWNTO 0);
		tx_datain		: IN STD_LOGIC_VECTOR (127 DOWNTO 0);
		tx_digitalreset		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		pll_locked		: OUT STD_LOGIC_VECTOR (0 DOWNTO 0);
		reconfig_fromgxb		: OUT STD_LOGIC_VECTOR (16 DOWNTO 0);
		rx_clkout		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_ctrldetect		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
		rx_dataout		: OUT STD_LOGIC_VECTOR (127 DOWNTO 0);
		rx_disperr		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
		rx_errdetect		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
		rx_freqlocked		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_patterndetect		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
		rx_phase_comp_fifo_error		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_pll_locked		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_syncstatus		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
		tx_clkout		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		tx_dataout		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		tx_phase_comp_fifo_error		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0)
	);
end component;
	component nios is
		port (
			avm_pod_address      : out std_logic_vector(15 downto 0);                    -- address
			avm_pod_read         : out std_logic;                                        -- read
			avm_pod_readdata     : in  std_logic_vector(31 downto 0) := (others => 'X'); -- readdata
			avm_pod_write        : out std_logic;                                        -- write
			avm_pod_writedata    : out std_logic_vector(31 downto 0);                    -- writedata
			avm_pod_waitrequest  : in  std_logic                     := 'X';             -- waitrequest
			avm_qsfp_address     : out std_logic_vector(15 downto 0);                    -- address
			avm_qsfp_read        : out std_logic;                                        -- read
			avm_qsfp_readdata    : in  std_logic_vector(31 downto 0) := (others => 'X'); -- readdata
			avm_qsfp_write       : out std_logic;                                        -- write
			avm_qsfp_writedata   : out std_logic_vector(31 downto 0);                    -- writedata
			avm_qsfp_waitrequest : in  std_logic                     := 'X';             -- waitrequest
			avm_sc_address       : out std_logic_vector(15 downto 0);                    -- address
			avm_sc_read          : out std_logic;                                        -- read
			avm_sc_readdata      : in  std_logic_vector(31 downto 0) := (others => 'X'); -- readdata
			avm_sc_write         : out std_logic;                                        -- write
			avm_sc_writedata     : out std_logic_vector(31 downto 0);                    -- writedata
			avm_sc_waitrequest   : in  std_logic                     := 'X';             -- waitrequest
			clk_clk              : in  std_logic                     := 'X';             -- clk
			i2c_sda_in           : in  std_logic                     := 'X';             -- sda_in
			i2c_scl_in           : in  std_logic                     := 'X';             -- scl_in
			i2c_sda_oe           : out std_logic;                                        -- sda_oe
			i2c_scl_oe           : out std_logic;                                        -- scl_oe
			pio_export           : out std_logic_vector(31 downto 0);                    -- export
			rst_reset_n          : in  std_logic                     := 'X';             -- reset_n
			sc_clk_clk           : in  std_logic                     := 'X';             -- clk
			sc_reset_reset_n     : in  std_logic                     := 'X';             -- reset_n
			spi_MISO             : in  std_logic                     := 'X';             -- MISO
			spi_MOSI             : out std_logic;                                        -- MOSI
			spi_SCLK             : out std_logic;                                        -- SCLK
			spi_SS_n             : out std_logic_vector(1 downto 0)                      -- SS_n
		);
	end component nios;

end package;
