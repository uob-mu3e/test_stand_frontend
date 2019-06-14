library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use ieee.math_real.all; -- for UNIFORM, TRUNC
use ieee.numeric_std.all; -- for TO_UNSIGNED
use ieee.std_logic_textio.all;	-- for write std_logic_vector to line
library std;
use std.textio.all;             --FOR LOGFILE WRITING
library mutrig_sim;
use mutrig_sim.txt_util.all;
library modelsim_lib;
use modelsim_lib.util.all;

--use work.mudaq_registers.all;
--use work.pcie_components.all;
--use work.mudaq_components.all;
--use work.mutrig_components.all;
--use work.devboard_components.all;
--use work.gbt_components.all;
--use work.lvds_components.all;

use work.daq_constants.all;
use work.mutrig_constants.all;

library mutrig_sim;
use mutrig_sim.txt_util.all;
use mutrig_sim.datapath_types.all;
use mutrig_sim.datapath_helpers.all;

entity testbench is
end testbench; 

architecture RTL of testbench is
--dut definition
component mutrig_datapath is
  generic(
	N_ASICS : integer :=1
  );
  port (
	i_rst			: in  std_logic;
	i_stic_txd		: in  std_logic_vector( N_ASICS-1 downto 0);-- serial data
	i_refclk_125		: in  std_logic;                 		-- ref clk for pll

	--interface to asic fifos
	i_clk_core		: in  std_logic; --fifo reading side clock
	o_fifo_empty		: out std_logic;
	o_fifo_data		: out std_logic_vector(35 downto 0);
	i_fifo_rd		: in  std_logic;
	--slow control
	i_SC_disable_dec	: in std_logic;
	i_SC_mask		: in std_logic_vector( N_ASICS-1 downto 0);
	i_SC_datagen_enable	: in std_logic;
	i_SC_datagen_shortmode	: in std_logic;
	i_SC_datagen_count	: in std_logic_vector(9 downto 0);
	--monitors
	o_receivers_usrclk	: out std_logic;              		-- pll output clock
	o_receivers_pll_lock	: out std_logic;			-- pll lock flag
	o_receivers_dpa_lock	: out std_logic_vector( N_ASICS-1 downto 0);			-- dpa lock flag per channel
	o_receivers_ready	: out std_logic_vector( N_ASICS-1 downto 0);-- receiver output ready flag
	o_frame_desync		: out std_logic;
	o_buffer_full		: out std_logic
  );
end component mutrig_datapath;

constant N_ASICS 	: natural := 4;
constant N_ASICS_MAX	: natural := 4;

-- ASIC signals --{{{
signal i_asic_tx_p	  	: std_logic_vector(N_ASICS-1 downto 0):=(others =>'0');
signal asic_clk_common	  	: std_logic:='0';
signal asic_clk		  	: std_logic_vector(N_ASICS_MAX-1 downto 0):=(others =>'0');
signal asics_rst	  	: std_logic:='0';
--SiPM INPUTS
signal a_sipm_in_p : std_logic_vector(N_ASICS*32-1 downto 0):=(others=>'0');	--POSITIVE POLARITY
signal a_sipm_in_n : std_logic_vector(N_ASICS*32-1 downto 0):=(others=>'0');	--NEGATIVE POLARITY
signal asics_sclk		: std_logic:='0';
signal asics_mosi		: std_logic:='0';
signal asics_miso		: std_logic;
signal asic_cs			: std_logic_vector(N_ASICS-1 downto 0):=(others => '1');
--}}}

----spi interface (not used, we just force the configuration vector)
--signal s_spi_clk 	: std_logic := '0';
--signal s_spi_clk_ena	: std_logic := '0';     --ENABLE SIGNAL FOR THE SPI CLOCK
--signal SPI_data             : STD_LOGIC_VECTOR(0 to N_CONF_BITS-1):=(others=>'0');         --THE DATA VECTOR TO BE SENT TO THE SPI SLAVE
--signal SPI_return_data      : STD_LOGIC_VECTOR(0 to N_CONF_BITS-1):=(others=>'0');
--signal config : stic3_spi_config;
--signal slave_config : stic3_spi_config;


--system signals receiver
signal i_rst		: std_logic:='0';
signal i_refclk_125	: std_logic:='0';
--receiver fifos
signal i_coreclk  	: std_logic:='0';

--monitoring
signal o_receivers_pll_clock	: std_logic;
signal o_receivers_pll_lock	: std_logic;
signal o_receivers_dpa_lock	: std_logic_vector(N_ASICS-1 downto 0);
signal o_receivers_ready	: std_logic_vector(N_ASICS-1 downto 0);
signal o_frame_desync		: std_logic;
--fifo interface
signal s_fifo_empty 		: std_logic:='0';
signal s_fifo_data		: std_logic_vector(35 downto 0);
signal s_fifo_rd		: std_logic;
signal s_fifo_rd_last		: std_logic;

signal i_SC_mask		: std_logic_vector(N_ASICS-1 downto 0);
begin

-- basic stimulus for receiver
i_coreclk	<= not i_coreclk after  5 ns;	-- 100 MHz system core clock (just different for fun)
i_refclk_125	<= not i_refclk_125 after  4 ns;	-- 125 MHz tranceiver reference clock
i_rst		<= '1' after 20 ns, '0' after 200 ns;	-- Basic reset of GBT

-- basic stimulus for asics
asic_clk_common			<= not asic_clk_common  after 0.8 ns;	--  625 MHz ASIC clock
asics_rst			<= '1' after 20 ns, '0' after 60 ns;	-- Basic reset of ASIC


asic_clk(0)	<= asic_clk_common  after 0.1 ns;
asic_clk(1)	<= asic_clk_common  after 0.2 ns;
asic_clk(2)	<= asic_clk_common  after 0.3 ns;
asic_clk(3)	<= asic_clk_common  after 0.4 ns;


i_SC_mask <= (3=>'1',  others=>'0');

--- mutrig digital part as main stimulus
gen_asic: for i in 0 to N_ASICS-1 generate
	asic: entity mutrig_sim.stic3_top  --{{{
	PORT MAP(
		a_sipm_in_p => a_sipm_in_p((i+1)*32-1 downto i*32),
		a_sipm_in_n => a_sipm_in_n((i+1)*32-1 downto i*32),

		--a_pll1_Ref_p => '0',--asic_clk(i),
		--a_pll1_Ref_n => '0',--asic_clk(i),
		a_pll1_mon => open,
		di_chip_rst => asics_rst,
		di_channel_rst => '0',
		di_fifo_ext_trig => '0',
		di_ser_clk_n => asic_clk(i),
		di_ser_clk_p => not asic_clk(i),
		do_txd_p => i_asic_tx_p(i),
		do_txd_n => open,
		a_dmon_p => open,
		a_dmon_n => open,
		a_amon => open,

			--SPI PINS	(DIGITAL BOTTOM PINS)
		di_sdi => asics_mosi,
		do_sdo => asics_miso,
		di_sclk => asics_sclk,
		di_cs => asic_cs(i),

		do_cec_sdo => open,
		di_cec_cs => '1',
		di_dummy => '0',
		do_dummy => open
	  );  --}}}

end generate;
--disconnect one asic
--i_asic_tx_p(3)<='0';

--dut
dut: mutrig_datapath
	generic map ( N_ASICS => N_ASICS )
	port map (
		i_rst			=> i_rst,
		i_stic_txd		=> i_asic_tx_p,
		i_refclk_125		=> i_refclk_125,
		o_receivers_usrclk	=> o_receivers_pll_clock,
		o_receivers_pll_lock	=> o_receivers_pll_lock,
		o_receivers_dpa_lock	=> o_receivers_dpa_lock,
		o_receivers_ready	=> o_receivers_ready,

		i_clk_core		=> i_coreclk,
    		o_fifo_empty		=> s_fifo_empty,
    		o_fifo_data		=> s_fifo_data,
    		i_fifo_rd		=> s_fifo_rd,
		i_SC_disable_dec	=> '1',				--disable decoder to speed up simulation
		i_SC_mask		=> i_SC_mask,
		o_frame_desync		=> o_frame_desync,
		i_SC_datagen_enable	=> '0',
		i_SC_datagen_shortmode	=> '0',
		i_SC_datagen_count	=> (3=>'1',others=>'0')
	);

---------------------------------------------------------------
-- fifo readout stimulus
fifo_ro: process (i_coreclk)
begin
	if (rising_edge(i_coreclk)) then
		if (i_rst='1') then s_fifo_rd <='0';
		else
			s_fifo_rd_last<=s_fifo_rd;
			if(s_fifo_rd='0') then
				if (s_fifo_empty='0') then 
					s_fifo_rd<= '1'; --read only when not empty
				else
					s_fifo_rd<= '0';
				end if;
			else
				s_fifo_rd <= '0';	 --read only every other cycle
			end if;
		end if;
	end if;
end process;

-- fifo readout logger
fifo_logging: process (i_coreclk)
file log_file : TEXT open write_mode is "readout_data.txt";
variable l : line;
begin
	if rising_edge(i_rst) then
		-- write header
	write(l, string'("--------------------------------------------------"));
		
	elsif (rising_edge(i_coreclk) and s_fifo_rd_last='1') then
		if(s_fifo_data(33 downto 32)="10") then
			write(log_file,"Frame Header "&
				HT & "RAW "& to_hstring(s_fifo_data) & 
				HT & "FSYN="& "0x" & to_hstring(s_fifo_data(16 downto 16)) &
				HT & "FID ="& "0x" & to_hstring(s_fifo_data(15 downto 0)) & 
				LF);
		elsif(s_fifo_data(33 downto 32)="11") then
			write(log_file,"Frame Trailer "&
				HT & "RAW "& to_hstring(s_fifo_data) & 
				HT & "L2F="& "0x" & to_hstring(s_fifo_data(1 downto 1)) &
				HT & "CRC="& "0x" & to_hstring(s_fifo_data(0 downto 0)) &
				LF);
		elsif(s_fifo_data(33 downto 32)="00") then
			write(log_file,"Hit data");
		end if;
		if(s_fifo_data(33 downto 33)="0") then
			write(log_file,
				HT & " RAW "& to_hstring(s_fifo_data) & 
				HT & "ASIC "& natural'image(to_integer(unsigned(s_fifo_data(31 downto 28)))) & 
				HT & "TYPE "& natural'image(to_integer(unsigned(s_fifo_data(27 downto 27)))) & 
				HT & "  CH "& natural'image(to_integer(unsigned(s_fifo_data(26 downto 22)))) & 
				HT & " EBH "& to_hstring(s_fifo_data(21 downto 21)) & 
				HT & " ECC "& to_hstring(s_fifo_data(20 downto  6)) & 
				HT & " EFC "& to_hstring(s_fifo_data( 5 downto  1)) & 
				HT & "EFLG "& to_hstring(s_fifo_data( 0 downto  0)) &
				LF);

		end if;

		flush(log_file);
	end if;
end process;



-- generate some hits, two per asic
gen_hits: process
begin	
	a_sipm_in_p <= (others => '0');
	wait for 20 us;
	while (TRUE) loop
		a_sipm_in_p(0) <= '1';
		--for i in 0 to N_ASICS-1 loop
		--	a_sipm_in_p(i*32+0) <= '1';
		--	a_sipm_in_p(i*32+1) <= '1';
		--end loop;
		wait for 100 ns;
		a_sipm_in_p <= (others => '0');
	wait for 6 us;
	end loop;
end process;

end architecture;
