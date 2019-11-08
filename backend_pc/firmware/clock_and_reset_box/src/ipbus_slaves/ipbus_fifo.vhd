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


-- ipbus_ctrlreg_v
--
-- Generic control / status register bank
--
-- Provides N_CTRL control registers (32b each), rw
-- Provides N_STAT status registers (32b each), ro
--
-- Bottom part of read address space is control, top is status
--
-- Dave Newbold, July 2012

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use work.ipbus.all;
use work.ipbus_reg_types.all;
Library UNISIM;
use UNISIM.vcomponents.all;

Library UNIMACRO;
use UNIMACRO.vcomponents.all;
entity ipbus_fifo is
	generic(
		rd_or_wr: boolean := false --false is a fifo which is written by the ipbus
	);
	port(
		ipbclk: in std_logic;
		extclk: in std_logic; 
		reset: in std_logic;
		ipbus_in: in ipb_wbus;
		ipbus_out: out ipb_rbus;
		fifo_RD: in std_logic;
		fifo_WR: in std_logic;
		fifo_DO: out std_logic_vector(31 downto 0);
		fifo_DI: in std_logic_vector(31 downto 0);
		WRCOUNT: out std_logic_vector(8 downto 0);
		RDCOUNT: out std_logic_vector(8 downto 0);
		empty: out std_logic
--		qmask: in ipb_reg_v(0 downto 0) := (others => (others => '1'))	
--		stb: out std_logic
	);
	
end ipbus_fifo;

architecture rtl of ipbus_fifo is

	constant ADDR_WIDTH: integer := 2; --for read and write

	signal reg: ipb_reg_v(0 downto 0);
	signal ri: ipb_reg_v(2 ** ADDR_WIDTH - 1 downto 0);
	signal sel: integer range 0 to 2 ** ADDR_WIDTH - 1 := 0;
	signal ALMOSTEMPTY : std_logic;
	signal ALMOSTFULL : std_logic;
	signal EMPTY_buf : std_logic;
	signal FULL : std_logic;
	--signal RDCOUNT : std_logic_vector (8 downto 0);
	signal RDERR : std_logic;
	--signal WRCOUNT : std_logic_vector (8 downto 0);
	signal WRERR : std_logic;
	signal DI : std_logic_vector(31 downto 0);
	signal RDEN : std_logic;
	signal WREN : std_logic;
	signal reset_fifo : std_logic;
	signal rd_b : std_logic;
	signal fifo_out : std_logic_vector(31 downto 0);
	signal ipbwrite_b,ipbread_b: std_logic;
	signal ipberr : std_logic;
	signal rdclk : std_logic;
	signal wrclk : std_logic;
	signal delayed_empty : std_logic;
	signal invoke_err: std_logic;

begin



    
	sel <= to_integer(unsigned(ipbus_in.ipb_addr(ADDR_WIDTH - 1 downto 0))) when ADDR_WIDTH > 0 else 0;

--			wr_rd_monitor: process(ipbclk)
--			begin
--				if rising_edge(ipbclk) then
--					if reset = '1' then
--						WREN <= '0';
--						ipbwrite_b <= '0';
--						RDEN <= '0';
--						ipbread_b <= '0';
--					--elsif ipbus_in.ipb_strobe = '1' and ipbus_in.ipb_write = '1' then
--					elsif sel = 0 and ipbus_in.ipb_write = '1'  and ipbus_in.ipb_strobe = '1' then
--						WREN <= '1';
--						RDEN <= '0';
--					elsif sel = 1 and ipbus_in.ipb_strobe = '1'  then
--						WREN <= '0';
--						RDEN <= '1';
--					else
--						WREN <= '0';
--						RDEN <= '0';
--					end if;
--					ipbwrite_b <= ipbus_in.ipb_write;
--					ipbwrite_b <= ipbus_in.ipb_strobe;
--				end if;
--			end process;
			

--	process(ipbclk)
--	begin
--		if rising_edge(ipbclk) then
--			for i in N_REG - 1 downto 0 loop
--				if sel = i then
--					stb <= ipbus_in.ipb_strobe and ipbus_in.ipb_write;
--				else
--					stb <= '0';
--				end if;
--			end loop;
--		end if;
--	end process;
	
--	ri(N_REG - 1 downto 0) <= reg;
--	ri(2 ** ADDR_WIDTH - 1 downto N_REG) <= (others => (others => '0'));
	fifo_DO <= fifo_out;
	ipbus_out.ipb_rdata <= fifo_out;
	ipbus_out.ipb_ack <= ipbus_in.ipb_strobe;
	ipbus_out.ipb_err <= ipberr;
	ipberr_assign: if rd_or_wr generate
--		rd_err : process (extclk, reset) is
--		begin
--			if reset = '1' then
--				ipberr <= '0';
--			elsif rising_edge(extclk) then
--				if EMPTY_buf = '0' then
--					ipberr <= '1';
--				else
--					ipberr <= '0';
--				end if;
--			end if;
--		end process rd_err;
		ipberr <= '1' when (ipbus_in.ipb_strobe = '1') and (invoke_err = '1') else '0'; --delaying the empty to allow the read of the last item in fifo before empty goes high
		rdclk <= ipbclk;
		wrclk <= extclk;
		WREN <= fifo_WR;
		--RDEN <= '1' when sel = 1 and ipbus_in.ipb_strobe = '1' else '0';
		RDEN <= ipbus_in.ipb_strobe;
		DI <= fifo_DI; --and qmask(sel);
		
		error_invoker : process (ipbclk, reset) is
		begin
			if reset = '1' then
				invoke_err <= '0';
			elsif rising_edge(ipbclk) then
				if  (ipbus_in.ipb_strobe = '1') and (EMPTY_buf = '1')  then
					invoke_err <= '1';
				elsif (EMPTY_buf = '0') then
					invoke_err <= '0';
				end if;
			end if;
		end process error_invoker;
		
--		
--		delay1 : FDCE
--		generic map (
--			INIT => '0') -- Initial value of register ('0' or '1')  
--		port map (
--			Q => delayed_empty,      -- Data output
--			C => ipbclk,      -- Clock input
--			CE => '1',    -- Clock enable input
--			CLR => '0',  -- Asynchronous clear input
--			D => EMPTY_buf       -- Data input
--		);	
		   
	else generate 
--		wr_err : process (extclk, reset) is
--		begin
--			if reset = '1' then
--				ipberr <= '0';
--			elsif rising_edge(extclk) then
--				if FULL = '0' then
--					ipberr <= '1';
--				else
--					ipberr <= '0';
--				end if;
--			end if;
--		end process wr_err;
		ipberr <= ipbus_in.ipb_strobe and FULL;
		rdclk <= extclk;
		wrclk <= ipbclk;
--		WREN <= '1' when sel = 0 and ipbus_in.ipb_write = '1'  and ipbus_in.ipb_strobe = '1' else '0';
		WREN <= '1' when ipbus_in.ipb_write = '1'  and ipbus_in.ipb_strobe = '1' else '0';
		RDEN <= fifo_RD;
		DI <= ipbus_in.ipb_wdata;
	end generate;
	empty <= EMPTY_buf;
	
	
	
	

	

   -- FIFO_DUALCLOCK_MACRO: Dual-Clock First-In, First-Out (FIFO) RAM Buffer
   --                       Kintex-7
   -- Xilinx HDL Language Template, version 2017.4

   -- Note -  This Unimacro model assumes the port directions to be "downto". 
   --         Simulation of this model with "to" in the port directions could lead to erroneous results.

   -----------------------------------------------------------------
   -- DATA_WIDTH | FIFO_SIZE | FIFO Depth | RDCOUNT/WRCOUNT Width --
   -- ===========|===========|============|=======================--
   --   37-72    |  "36Kb"   |     512    |         9-bit         --
   --   19-36    |  "36Kb"   |    1024    |        10-bit         --
   --   19-36    |  "18Kb"   |     512    |         9-bit         --
   --   10-18    |  "36Kb"   |    2048    |        11-bit         --
   --   10-18    |  "18Kb"   |    1024    |        10-bit         --
   --    5-9     |  "36Kb"   |    4096    |        12-bit         --
   --    5-9     |  "18Kb"   |    2048    |        11-bit         --
   --    1-4     |  "36Kb"   |    8192    |        13-bit         --
   --    1-4     |  "18Kb"   |    4096    |        12-bit         --
   -----------------------------------------------------------------

   FIFO_DUALCLOCK_MACRO_inst : FIFO_DUALCLOCK_MACRO
   generic map (
      DEVICE => "7SERIES",            -- Target Device: "VIRTEX5", "VIRTEX6", "7SERIES" 
      ALMOST_FULL_OFFSET => X"0080",  -- Sets almost full threshold
      ALMOST_EMPTY_OFFSET => X"0080", -- Sets the almost empty threshold
      DATA_WIDTH => 32,   -- Valid values are 1-72 (37-72 only valid when FIFO_SIZE="36Kb")
      FIFO_SIZE => "18Kb",            -- Target BRAM, "18Kb" or "36Kb" 
      FIRST_WORD_FALL_THROUGH => FALSE) -- Sets the FIFO FWFT to TRUE or FALSE
   port map (
      ALMOSTEMPTY => ALMOSTEMPTY,   -- 1-bit output almost empty
      ALMOSTFULL => ALMOSTFULL,     -- 1-bit output almost full
      DO => fifo_out,                     -- Output data, width defined by DATA_WIDTH parameter
      EMPTY => EMPTY_buf,               -- 1-bit output empty
      FULL => FULL,                 -- 1-bit output full
      RDCOUNT => RDCOUNT,           -- Output read count, width determined by FIFO depth
      RDERR => RDERR,               -- 1-bit output read error
      WRCOUNT => WRCOUNT,           -- Output write count, width determined by FIFO depth
      WRERR => WRERR,               -- 1-bit output write error
      DI => DI,                     -- Input data, width defined by DATA_WIDTH parameter
      RDCLK => rdclk,               -- 1-bit input read clock
      RDEN => RDEN,                 -- 1-bit input read enable
      RST => reset_fifo,                   -- 1-bit input reset
      WRCLK => wrclk,               -- 1-bit input write clock
      WREN => WREN                  -- 1-bit input write enable
   );
   -- End of FIFO_DUALCLOCK_MACRO_inst instantiation
	
	--RDEN <= fifo_RD and (not reset);	-- to make sure the RD pulses remain low when reset is high
	
	
	fifo_reset: entity work.fifo_reset_ctrl
	port map(
		clk  => ipbclk,
		rsti => reset,
		rsto => reset_fifo
	);
--   monitor_rd : process (rdclk) is
--   begin
--   	if rising_edge(rdclk) then
--   		if reset_fifo = '1' then
--   			RDEN <= '0';
--   			rd_b <= '0';
--   		else
--   			if (fifo_RD = '1'and rd_b = '0') then
--   				RDEN <= '1';
--   			else
--   				RDEN <= '0';
--   			end if;
--   			rd_b <= fifo_RD;
--   		end if;
--   	end if;
--   end process monitor_rd;
   

   	
end rtl;
