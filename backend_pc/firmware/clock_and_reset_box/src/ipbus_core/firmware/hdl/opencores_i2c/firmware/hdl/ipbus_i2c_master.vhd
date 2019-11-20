-- ipbus_i2c_master
--
-- Wrapper for opencores i2c wishbone slave
--
-- Dave Newbold, Jan 2012

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.ipbus.all;

entity ipbus_i2c_master is
	generic(addr_width: natural := 0);
	port(
		clk: in std_logic;
		rst: in std_logic;
		ipbus_in: in ipb_wbus;
		ipbus_out: out ipb_rbus;
		scl_o: out std_logic;
		scl_i: in std_logic;
		sda_o: out std_logic;
		sda_i: in std_logic
	);
	
end ipbus_i2c_master;

architecture rtl of ipbus_i2c_master is

	signal stb, ack, sda_enb: std_logic;

begin

	stb <= ipbus_in.ipb_strobe and not ack;

	i2c: entity work.i2c_master_top port map(
		wb_clk_i => clk,
		wb_rst_i => rst,
		arst_i => '1',
		wb_adr_i => ipbus_in.ipb_addr(2 downto 0),
		wb_dat_i => ipbus_in.ipb_wdata(7 downto 0),
		wb_dat_o => ipbus_out.ipb_rdata(7 downto 0),
		wb_we_i => ipbus_in.ipb_write,
		wb_stb_i => stb,
		wb_cyc_i => '1',
		wb_ack_o => ack,
		scl_pad_i => scl_i,
		scl_padoen_o => scl_o,
		sda_pad_i => sda_i,
		sda_padoen_o => sda_o
	);
	
	ipbus_out.ipb_rdata(31 downto 8) <= (others => '0');
	ipbus_out.ipb_ack <= ack;
	ipbus_out.ipb_err <= '0';
	
	--scl <= scl_i;
	
end rtl;
