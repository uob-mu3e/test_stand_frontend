-- hit counter: cec
-- copied (small modifications) from: 
------------------------------------------------------------------------
-- File name 	: cec_spi_slave.vhd
-- Author	: Huangshan Chen
-- Description	:
--   1. Read the channel event counter at time intervals
--   2. put the data to sys_clk domain for reading out
--
--
--  (the synchronous fsm is sync to the falling edge of sclk)
--  sclk:       --|__|--|__|--|__ ... __|--|__|--|__
--  sclk_en:    __|-------------- ... -----|________
--  o_sclk:     _____|--|__|--|__ ... __|--|________
--  cs_l:       --------|________ ...____________|--
--                   A |B |C |D | ...   E |F |
--
--  A: cec spi master latching data (chip side)
--  B: cec spi slave latching X (FPGA side), since the output of master is not
--     enabled yet and the master need 1 more clk cycle to put the 1st bit to the
--     output...
--  C: cec spi master shifting data, output reg. latching 1st bit(chip side)
--  D: cec spi slave latching 1nd bit (FPGA side)
--  ...
--  E: cec spi master output reg. latching last bit (chip side)
--  F: cec spi slave latching last bit (FPGA side),
--     cs_l keeps low such thar the output of master is enabled.
--
--
-- The channel event counter structure for MuTRiGv1:
-- |<-- MSB(423) -- ... -- LSB(0) -->|
-- | header("00011100") | event_count_ch31 (1 bit overflow + 12 bit counter value) | event_count_ch30 | ... | event_count_ch0 |
--
------------------------------------------------------------------------

library IEEE;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity spi_hitcounter is
generic(
	C_NUM_CNT_BITS		: integer := 32*13 + 8;	-- length of CEC data register		
	C_TRANS_INTER		: integer := 2000;	-- the factor between sclk(2MHz) and the reading frenquency, e.g. 2MHz/2000 = 1KHz
	C_NUM_CLIENTS		: integer :=  1;
	C_HANDLEID_WIDTH 	: integer :=  1
);
port(
	i_clk			: in  std_logic;                                     -- system clock signal (125 MHz?)
	i_rst			: in  std_logic;                                     -- reset
	i_handleid 	   	: in  std_logic_vector(C_HANDLEID_WIDTH-1 downto 0); -- handle id of chip, will only talk to this one
	o_eot			: out std_logic;                                     -- end of transmission for daq controle
	o_hitcounter		: out std_logic_vector(C_NUM_CNT_BITS-1 downto 0);     -- 52 byte, 8 32bit words

	-- spi signals
	o_sclk			: out std_logic;                                     -- common clock  to all connected spi clients
	i_mdi			: in  std_logic_vector(C_NUM_CLIENTS-1 downto 0);	 -- individual spi slave input
	o_cs			: out std_logic_vector(C_NUM_CLIENTS-1 downto 0)	 -- individual chip select signals
);
end spi_hitcounter;

architecture RTL of spi_hitcounter is
component cec_clk
PORT ( 
	inclk0	: IN STD_LOGIC  := '0';  -- 125MHz base clock
	c0	: OUT STD_LOGIC          --   2MHz spi clock
);
end component;


-- clocks and resets 
signal s_sclk, s_rst		: std_logic;     -- 2MHz clock, rst sync to s_sclk
signal s_sclk_en, s_cs_l	: std_logic;
signal s_mdi			: std_logic;
-- state machine
type fs_states is (
	FS_IDLE,
	FS_START,
	FS_TRANS,
	FS_EOT
);
signal s_state : fs_states; 
	
-- comands
signal s_start_trans, s_cec_ready : std_logic;

signal s_hitcounter	: std_logic_vector(C_NUM_CNT_BITS-1 downto 0); -- 416 bits, 52 bytes 
signal s_bit_cnt	: integer range 0 to C_TRANS_INTER-1 := 0;
		  
begin
s_mdi <= i_mdi(to_integer(unsigned(i_handleid)));
gen_csn: process (i_handleid, s_cs_l)
begin
	o_cs<=(others=>'1');
	o_cs(to_integer(unsigned(i_handleid)))<=s_cs_l;
end process;
o_sclk <= s_sclk and s_sclk_en;

u_cec_clk : cec_clk 
	port map (
		inclk0	=> i_clk,  -- 125MHz
		c0	=> s_sclk  --   2Mhz
	);

-- start transition process--
-- generate 1 sclk pulse every C_TRANS_INTER sclk cycle
proc_start_trans : process(s_sclk) -- 2 MHz
variable cnt : integer range 0 to C_TRANS_INTER-1 := 0;
begin
	if falling_edge(s_sclk) then
		if s_rst = '1' then
			cnt := C_TRANS_INTER-1;
		elsif cnt = 0 then
			cnt := C_TRANS_INTER-1;
			s_start_trans <= '1';
		else
			cnt := cnt -1;
			s_start_trans <= '0';
		end if;
	end if;
end process;


-- process of taking data
proc_trans_data : process(s_sclk)
begin
	if falling_edge(s_sclk) then
		if s_rst = '1' then
			s_state 	<= FS_IDLE;
			s_sclk_en 	<= '0';
			s_cs_l 		<= '1'; -- active low
			s_cec_ready <= '0';
			s_bit_cnt 	<= C_NUM_CNT_BITS-1;
		else
		case s_state is
			when FS_IDLE =>
				s_sclk_en	<= '0';
				s_cs_l		<= '1';
				s_cec_ready	<= '0';
				s_bit_cnt	<= C_NUM_CNT_BITS -1;
				if s_start_trans = '1' then
					s_state	<= FS_START;
				end if;

			when FS_START => -- make s_sclk_en = '1' one clk cycle earlier than cs_l, for master to latch the data
				s_sclk_en	<= '1';
				s_cs_l		<= '1';
				s_cec_ready	<= '0';
				s_bit_cnt	<= C_NUM_CNT_BITS -1;
				s_state		<= FS_TRANS;

			when FS_TRANS =>
				s_sclk_en	<= '1';
				s_cs_l		<= '0';
				s_cec_ready	<= '0';
				s_bit_cnt	<= s_bit_cnt -1;
				s_hitcounter	<= s_hitcounter(s_hitcounter'high-1 downto 0) & s_mdi; -- shift data into s_cec_data, 1st bit is 'X' since the i_cec_mosi is not ready
				if s_bit_cnt = 0 then
					s_state <= FS_EOT;
				end if;

			when FS_EOT => -- keep cs_l ='0' for one more clk cyclt to latch the last bit
				s_sclk_en	<= '0';
				s_cs_l		<= '0';
				s_bit_cnt	<= 0;
				s_hitcounter	<= s_hitcounter(s_hitcounter'high-1 downto 0) & s_mdi; -- shift data into s_cec_data
				s_cec_ready	<= '1';
				s_state		<= FS_IDLE;

			when others =>
				s_state <= FS_IDLE;
		end case;
		end if;
	end if;
end process;

-- sync the s_hitcounter to 125 MHz clk
-- not really necessary since s_sclk falling edge is sync to the rising edge of the 125 MHz clk
-- keep it here just to be safe in timing
proc_latching_output : process(i_clk)
begin
	if rising_edge(i_clk) then
		--if s_cs_l = '1' then
		o_hitcounter <= s_hitcounter;
		o_eot        <= s_cec_ready;
		--end if;
	end if;
end process;

-- sync reset to s_sclk falling edge
sync_reset : process(s_sclk)
begin
	if falling_edge(s_sclk) then
		s_rst 	<= i_rst;
	end if;
end process;
end;
