---------------------------------------
--
-- On the fly PRBS decoder for mutrig/stic event data
-- Latency two CC.
-- data (or more precisely data_valid flag) is only passed after decoder memory has been initialized, 
-- Initialization takes ~32k CC.
-- Frame header and trailer tags are passed as is, word after header (supposed to be second part of header payload) is passed as is
-- Konrad Briggl May 2019
-- 
-- konrad.briggl@unige.ch
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

entity prbs_decoder is
port (
--system
	i_coreclk	: in  std_logic;
	i_rst		: in  std_logic;
--data stream input
	i_data	: in std_logic_vector(33 downto 0);
	i_valid	: in std_logic;
--data stream output
	o_data	: out std_logic_vector(33 downto 0);
	o_valid	: out std_logic;
--disable block (make transparent)
	i_SC_disable_dec : in std_logic
);
end prbs_decoder;

architecture impl of prbs_decoder is
	component decoder_mem
		PORT
		(
			address		: IN STD_LOGIC_VECTOR (14 DOWNTO 0);
			clock		: IN STD_LOGIC  := '1';
			data		: IN STD_LOGIC_VECTOR (14 DOWNTO 0);
			wren		: IN STD_LOGIC  := '0';
			q		: OUT STD_LOGIC_VECTOR (14 DOWNTO 0)
		);
	end component;
	signal s_init : std_logic; --state vector, '1': Block is initializing
	signal s_init_prbs : std_logic_vector(14 downto 0); --initialization vector(original data)
	signal s_init_dec  : std_logic_vector(14 downto 0); --initialization vector(decoded data)
	signal s_addr_a,n_addr_a	: std_logic_vector(14 downto 0);    --r/w address on port a (init and T decoding)
	signal s_valid_2 : std_logic;
	signal s_valid_1 : std_logic;
	signal s_data_bypass_2  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_data_bypass_1  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_data_bypass    : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_data_dec,s_init_dec_d : std_logic_vector(14 downto 0); -- decoded data E timestamp
	signal n_select_bypass : std_logic;
	signal s_select_bypass : std_logic;
	signal n_is_header : std_logic;
	signal s_is_header : std_logic;
	signal l_is_header : std_logic;
begin


--decoder memory
--u_mem: decoder_mem -- replaced by direct implementation 
u_mem : altsyncram
GENERIC MAP (
	clock_enable_input_a => "BYPASS",
	clock_enable_output_a => "BYPASS",
	intended_device_family => "Stratix IV",
	lpm_hint => "ENABLE_RUNTIME_MOD=NO",
	lpm_type => "altsyncram",
	numwords_a => 32768,
	operation_mode => "SINGLE_PORT",
	outdata_aclr_a => "NONE",
	outdata_reg_a => "CLOCK0",
	power_up_uninitialized => "FALSE",
	ram_block_type => "M9K",
	read_during_write_mode_port_a => "NEW_DATA_NO_NBE_READ",
	widthad_a => 15,
	width_a => 15,
	width_byteena_a => 1
)
PORT MAP (
	address_a => s_addr_a,
	clock0 => i_coreclk,
	data_a => s_init_dec_d,
	wren_a => s_init,
	q_a => s_data_dec
);

--address assignments: PRBS data for TCC and ECC or initialization vector
--pipelined to improve timing budget
n_addr_a <= s_init_prbs when s_init='1' else i_data(20 downto 6);


--decoder init and bypass registers
p_sync: process(i_coreclk)
begin
	if rising_edge(i_coreclk) then
		--ram address and data input, data delay is only to compensate address pipelining.
		s_addr_a <= n_addr_a;
		s_init_dec_d <= s_init_dec;
		--memory initialisation state machine, data bypass control
		if(i_rst='1') then
			s_init_prbs <= (0=>'1', others=>'0');
			s_init_dec  <= (0=>'1', others=>'0');
			s_init <='1';
			s_valid_2<='0';
			s_valid_1<='0';
			o_valid<='0';
		else
			s_valid_2<= i_valid and (not s_init or i_SC_disable_dec);
			s_valid_1<= s_valid_2;
			o_valid<=s_valid_1 and not s_init;
			s_data_bypass_2 <= i_data;
			s_data_bypass_1 <= s_data_bypass_2;
			s_data_bypass <= s_data_bypass_1;

			s_init_prbs<=s_init_prbs(13 downto 0) & not (s_init_prbs(14) xor s_init_prbs(13));
			--s_init_prbs<=std_logic_vector(unsigned(s_init_prbs)+1);
			s_init_dec<=std_logic_vector(unsigned(s_init_dec)+1);
			if(s_init_dec=(0 to s_init_dec'length-1 =>'0')) then
				s_init<='0';
			end if;
			--scan input data for header or trailer, to be bypassed
			n_select_bypass <=i_data(33);
			if(i_data(33 downto 32)="10") then
				n_is_header <='1';
			else
				n_is_header <='0';
			end if;
			if(i_SC_disable_dec='1') then n_select_bypass <= '1'; end if;
			s_select_bypass<= n_select_bypass;
			s_is_header <=n_is_header;
			l_is_header <=s_is_header;
		end if;
	end if;
end process;

--output assignment: replace TCC and ECC by their decoded counterparts when not bypassing
o_data<=s_data_bypass	when s_select_bypass='1' or l_is_header='1' else
	s_data_bypass(33 downto 21) & s_data_dec(14 downto 0) & s_data_bypass(5 downto 0);
end architecture;

