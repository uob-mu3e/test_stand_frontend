---------------------------------------------------------------------------------
--
--   Copyright 2017 - Rutherford Appleton Laboratory and University of Bristol
--
--   Licensed under the Apache License, Version 2.0 (the "License");
--   you may not use this file except in compliance with the License.
--   You may obtain a copy of the License at
--
--       http://www.apache.org/licenses/LICENSE-2.0
--
--   Unless required by applicable law or agreed to in writing, software
--   distributed under the License is distributed on an "AS IS" BASIS,
--   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--   See the License for the specific language governing permissions and
--   limitations under the License.
--
--                                     - - -
--
--   Additional information about ipbus-firmare and the list of ipbus-firmware
--   contacts are available at
--
--       https://ipbus.web.cern.ch/ipbus
--
---------------------------------------------------------------------------------


-- ipbus_example
--
-- selection of different IPBus slaves without actual function,
-- just for performance evaluation of the IPbus/uhal system
--
-- Kristian Harder, March 2014
-- based on code by Dave Newbold, February 2011

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.ipbus.all;
use work.ipbus_reg_types.all;
use work.ipbus_decode_mu3e_address.all;
Library UNISIM;
use UNISIM.vcomponents.all;

entity ipbus_mu3e is
	port(
		ipb_clk: in std_logic;
		ipb_rst: in std_logic;
		ipb_in: in ipb_wbus;
		ipb_out: out ipb_rbus;
		nuke: out std_logic;
		soft_rst: out std_logic;
		clk_rd: in std_logic;
		clk_wr: in std_logic;
		fifo_DO: out std_logic_vector(31 downto 0);
		--in_cmd: in std_logic_vector(8 downto 0);
		--in_cmd: in std_logic_vector(31 downto 0);
		--fifo_rd: in std_logic;
		fifo_wr: in std_logic;
		fifo_DI: in std_logic_vector(31 downto 0);
		charisk: out std_logic_vector(3 downto 0);
		calibrate: out std_logic;
		data_valid: in std_logic;
		fan_current: in std_logic_vector(15 downto 0);
		mgt_adrs: out std_logic_vector(2 downto 0);
		mgt_mask: out std_logic_vector(7 downto 0);
		ic_control: out std_logic_vector(5 downto 0);
		sda_o: out std_logic;
		sda_i: in std_logic;
		scl_o: out std_logic;
		scl_i: in std_logic
		--bytecount: out std_logic_vector(1 downto 0)
	);

end ipbus_mu3e;



architecture rtl of ipbus_mu3e is
	signal calibratex: std_logic;
	signal ipbw: ipb_wbus_array(N_SLAVES - 1 downto 0);
	signal ipbr: ipb_rbus_array(N_SLAVES - 1 downto 0);
	signal fifo_out_empty : std_logic;
	signal charisk_empty : std_logic;
	--signal fifo_out : std_logic_vector(31 downto 0);
	signal ctrl_mu3e : ipb_reg_v(0 downto 0);
	signal ctrlmask : ipb_reg_v(0 downto 0);
	signal fifo_DOx : std_logic_vector(31 downto 0);
	signal delayed_empty : std_logic;
	signal fifo_out_WRCOUNT : std_logic_vector(8 downto 0);
	signal charisk_WRCOUNT : std_logic_vector(8 downto 0);
	signal data_ready: std_logic;
	signal fifo_rd,fifo_rd_d : std_logic;
	signal fifoa_out : std_logic_vector(31 downto 0);
	signal fifob_out : std_logic_vector(31 downto 0);
	signal charisk_RDCOUNT : std_logic_vector(8 downto 0);
	signal fifo_out_RDCOUNT : std_logic_vector(8 downto 0);
	signal chariskx : std_logic_vector(3 downto 0);
	--signal fifo_DO_probe : std_logic_vector(31 downto 0);
	signal delayed_data_ready : std_ulogic;
	signal delayed_fifo_out_empty : std_ulogic;
	signal delayed_charisk_empty : std_ulogic;
	signal stat_mu3e : ipb_reg_v(0 downto 0);
	signal soft_rstx : std_logic;
	signal isda_o,isda_i,iscl: std_logic;
	signal iscl_o : std_logic;
	signal iscl_i : std_logic;
	--signal chariskx : std_logic;
begin

-- ipbus address decode
		
	fabric: entity work.ipbus_fabric_sel
    generic map(
    	NSLV => N_SLAVES,
    	SEL_WIDTH => IPBUS_SEL_WIDTH)
    port map(
      ipb_in => ipb_in,
      ipb_out => ipb_out,
      sel => ipbus_sel_mu3e_address(ipb_in.ipb_addr),
      ipb_to_slaves => ipbw,
      ipb_from_slaves => ipbr
    );

-- Slave 0: id / rst reg

	slave0: entity work.ipbus_ctrlreg_v
		generic map(
			N_CTRL => 1,
			N_STAT => 1
		)
		port map(
			clk       => ipb_clk,
			reset     => ipb_rst,
			ipbus_in  => ipbw(N_SLV_CTRL_REG),
			ipbus_out => ipbr(N_SLV_CTRL_REG),
			d         => stat_mu3e,
			q         => ctrl_mu3e,
			qmask     => ctrlmask
		);
		ctrlmask(0) <= X"000fffff";
		stat_mu3e(0)(31 downto 16) <= fan_current;
		stat_mu3e(0)(15 downto 4) <= X"000";
		stat_mu3e(0)(3 downto 0) <= "000" & data_valid;
		soft_rstx <= ctrl_mu3e(0)(0);
		soft_rst <= soft_rstx;
		calibratex <= ctrl_mu3e(0)(1);
		calibrate <= calibratex;
		nuke <= ctrl_mu3e(0)(2);
		mgt_mask <= ctrl_mu3e(0)(10 downto 3);
		mgt_adrs <= ctrl_mu3e(0)(13 downto 11);
		ic_control <= ctrl_mu3e(0)(19 downto 14);
		
		slave1: entity work.ipbus_fifo
				generic map (
					rd_or_wr => true
				)
			port map(
				ipbclk => ipb_clk,
				extclk => clk_wr,
				reset => ipb_rst,
				ipbus_in => ipbw(N_SLV_FIFO_REG_IN),
				ipbus_out => ipbr(N_SLV_FIFO_REG_IN),
				fifo_RD => '0',
				fifo_WR => fifo_wr,
				fifo_DO => open,
				fifo_DI => fifo_DI,
				empty => open
			);
		
		slave2a: entity work.ipbus_fifo
				generic map (
					rd_or_wr => false
				)
			port map(
				ipbclk => ipb_clk,
				extclk => clk_rd,
				reset => ipb_rst,
				ipbus_in => ipbw(N_SLV_FIFO_REG_OUT),
				ipbus_out => ipbr(N_SLV_FIFO_REG_OUT),
				fifo_RD => fifo_rd,
				fifo_WR => '0',
				fifo_DO => fifoa_out,
				fifo_DI => X"00000000",
				WRCOUNT => fifo_out_WRCOUNT,
				RDCOUNT => fifo_out_RDCOUNT,
				empty => fifo_out_empty
			);
		slave2b: entity work.ipbus_fifo
				generic map (
					rd_or_wr => false
				)
			port map(
				ipbclk => ipb_clk,
				extclk => clk_rd,
				reset => ipb_rst,
				ipbus_in => ipbw(N_SLV_FIFO_REG_OUT_CHARISK),
				ipbus_out => ipbr(N_SLV_FIFO_REG_OUT_CHARISK),
				fifo_RD => fifo_rd,
				fifo_WR => '0',
				fifo_DO => fifob_out,
				fifo_DI => X"00000000",
				WRCOUNT => charisk_WRCOUNT,
				RDCOUNT => charisk_RDCOUNT,
				empty => charisk_empty
			);
			
--		fifo_output_mon : process (clk_rd, ipb_rst) is
--		begin
--			if ipb_rst = '1' then
--				fifo_DOx <= X"07bcbcbc";
--			elsif rising_edge(clk_rd) then
--				if (delayed_empty = '0') then 
--					fifo_DOx <= fifo_out;
--				else
--					fifo_DOx <= X"07bcbcbc";
--				end if;
--			end if;
--		end process fifo_output_mon;
		
		slave_i2c: entity work.ipbus_i2c_master
			port map(
				clk => ipb_clk,
				rst => ipb_rst,
				ipbus_in => ipbw(N_SLV_I2C),
				ipbus_out => ipbr(N_SLV_I2C),
			    ipbus_in_fast => ipbw(N_SLV_I2C_FAST),
                ipbus_out_fast => ipbr(N_SLV_I2C_FAST),
                ipbus_in_mem => ipbw(N_SLV_I2C_MEM),
                ipbus_out_mem => ipbr(N_SLV_I2C_MEM),
				scl_o => iscl_o,
				scl_i => iscl_i,
				sda_o => isda_o,
				sda_i => isda_i
			);
			
			
			ila1: entity work.ila_0
        port map( 
    clk  => ipb_clk,--: in STD_LOGIC;
    probe0(0) => ipbw(N_SLV_I2C).ipb_strobe,
    probe1(0) => ipbw(N_SLV_I2C).ipb_write,
    probe2    => ipbw(N_SLV_I2C).ipb_addr,
    probe3(0) => ipbw(N_SLV_I2C_FAST).ipb_strobe,
    probe4(0) => ipbw(N_SLV_I2C_FAST).ipb_write,
    probe5    => ipbw(N_SLV_I2C_FAST).ipb_addr,
    probe6(0) => ipbr(N_SLV_I2C).ipb_ack,
    probe7(0) => ipbr(N_SLV_I2C).ipb_err,
    probe8    => ipbr(N_SLV_I2C).ipb_rdata,
    probe9(0) => ipbr(N_SLV_I2C_FAST).ipb_ack,
    probe10(0) => ipbr(N_SLV_I2C_FAST).ipb_err,
    probe11    => ipbr(N_SLV_I2C_FAST).ipb_rdata,
    probe12(0) => iscl_o,
    probe13(0) => iscl_i,
    probe14(0) => isda_o,
    probe15(0) => isda_i 
 );
			
			
		   delay1 : FDCE
		   generic map (
		      INIT => '0') -- Initial value of register ('0' or '1')  
		   port map (
		      Q => delayed_charisk_empty,      -- Data output
		      C => clk_rd,      -- Clock input
		      CE => '1',    -- Clock enable input
		      CLR => '0',  -- Asynchronous clear input
		      D => charisk_empty       -- Data input
		   );	
		   delay2 : FDCE
		   generic map (
		      INIT => '0') -- Initial value of register ('0' or '1')  
		   port map (
		      Q => delayed_fifo_out_empty,      -- Data output
		      C => clk_rd,      -- Clock input
		      CE => '1',    -- Clock enable input
		      CLR => '0',  -- Asynchronous clear input
		      D => fifo_out_empty       -- Data input
		   );	

		fifo_DO <= fifo_DOx; --when (fifo_rd='0') else fifoa_out;
		charisk <= chariskx; --when (fifo_rd='0') else fifob_out(3 downto 0);
		
		process (clk_rd, ipb_rst) is
		begin
			if ipb_rst = '1' then
				fifo_DOx <= (others => '0');
				chariskx <= (others => '0');
				fifo_rd_d <= '0';
			elsif rising_edge(clk_rd) then
				if (fifo_rd='1' and fifo_rd_d = '1') then
					fifo_DOx <= fifoa_out;
					chariskx <= fifob_out(3 downto 0);
				elsif (calibratex='1') then
					fifo_DOx <= X"bcbc00bc";
					chariskx <= X"d";					
				else
					fifo_DOx <= X"bcbcbcbc";
					chariskx <= X"f";					
				end if;
				fifo_rd_d <= fifo_rd;
			end if;
		end process;
		
		data_ready_process : process (clk_rd,ipb_rst) is
		begin
			if ipb_rst = '1' then
				data_ready <= '0';
				
			elsif rising_edge(clk_rd) then
				--data_ready <= charisk_empty nor fifo_out_empty;
				if (fifo_out_WRCOUNT = charisk_WRCOUNT) then
					data_ready <= '1';
				else
					data_ready <= '0';
				end if;
				
			end if;
		end process data_ready_process;
		fifo_rd <= data_ready and (delayed_charisk_empty nor delayed_fifo_out_empty);
		
		--charisk <= X"b" when delayed_empty = '0' else X"f";
---- Slave 2: 1kword RAM
--
--	slave4: entity work.ipbus_ram
--		generic map(ADDR_WIDTH => 10)
--		port map(
--			clk => ipb_clk,
--			reset => ipb_rst,
--			ipbus_in => ipbw(N_SLV_RAM),
--			ipbus_out => ipbr(N_SLV_RAM)
--		);
--	
---- Slave 3: peephole RAM
--
--	slave5: entity work.ipbus_peephole_ram
--		generic map(ADDR_WIDTH => 10)
--		port map(
--			clk => ipb_clk,
--			reset => ipb_rst,
--			ipbus_in => ipbw(N_SLV_PRAM),
--			ipbus_out => ipbr(N_SLV_PRAM)
--		);





sda_o <= isda_o;
isda_i <= sda_i;
iscl_i <= scl_i;
scl_o <= iscl_o;


end rtl;