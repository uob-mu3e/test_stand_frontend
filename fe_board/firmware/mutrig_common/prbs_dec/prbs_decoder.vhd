---------------------------------------
--
-- On the fly PRBS decoder for mutrig/stic event data
-- Latency two CC.
-- data (or more precisely data_valid flag) is only passed after decoder memory has been initialized, 
-- Initialization takes ~32k CC.
-- Frame header and trailer tags are passed as is
-- Konrad Briggl May 2019
-- 
-- konrad.briggl@unige.ch
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

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
			address_a	: IN STD_LOGIC_VECTOR (14 DOWNTO 0);
			address_b	: IN STD_LOGIC_VECTOR (14 DOWNTO 0);
			clock		: IN STD_LOGIC  := '1';
			data_a		: IN STD_LOGIC_VECTOR (15 DOWNTO 0);
			data_b		: IN STD_LOGIC_VECTOR (15 DOWNTO 0);
			wren_a		: IN STD_LOGIC  := '0';
			wren_b		: IN STD_LOGIC  := '0';
			q_a		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
			q_b		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0)
		);
	end component;
	signal s_init : std_logic; --state vector, '1': Block is initializing
	signal s_init_prbs : std_logic_vector(14 downto 0); --initialization vector(original data)
	signal s_init_dec  : std_logic_vector(14 downto 0); --initialization vector(decoded data)
	signal s_addr_a	: std_logic_vector(14 downto 0);    --r/w address on port a (init and T decoding)
	signal s_addr_b	: std_logic_vector(14 downto 0);    --r/w address on port b (E decoding)
	signal s_valid_1 : std_logic;
	signal s_data_bypass_1  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_data_bypass    : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_data_dec : std_logic_vector(15 downto 0); -- decoded data E timestamp
	signal s_select_bypass_1 : std_logic;
	signal s_select_bypass : std_logic;
begin

--address assignments: PRBS data for TCC and ECC or initialization vector
s_addr_a <= s_init_prbs; -- when s_init='1' else i_data(20 downto 6);
s_addr_b <= i_data(20 downto 6);

--decoder memory
u_mem: decoder_mem
port map(
	address_a	=> s_addr_a,
	address_b	=> s_addr_b,
	clock		=> i_coreclk,
	data_a		=> "0" & s_init_dec,
	data_b		=> (others=>'0'),
	wren_a		=> s_init,
	wren_b		=> '0',
	q_a		=> open,
	q_b		=> s_data_dec 
);
--decoder init and bypass registers
p_sync: process(i_coreclk)
begin
	if rising_edge(i_coreclk) then
		if(i_rst='1') then
			s_init_prbs <= (0=>'1', others=>'0');
			s_init_dec  <= (0=>'1', others=>'0');
			s_init <='1';
			s_valid_1<='0';
			o_valid<='0';
		else
			s_valid_1<= i_valid and (not s_init or i_SC_disable_dec);
			o_valid<=s_valid_1;
			s_data_bypass_1 <= i_data;
			s_data_bypass <= s_data_bypass_1;

			s_init_prbs<=s_init_prbs(13 downto 0) & not (s_init_prbs(14) xor s_init_prbs(13));
			--s_init_prbs<=std_logic_vector(unsigned(s_init_prbs)+1);
			s_init_dec<=std_logic_vector(unsigned(s_init_dec)+1);
			if(s_init_dec=(0 to s_init_dec'length-1 =>'0')) then
				s_init<='0';
			end if;
			--scan input data for header or trailer, to be bypassed
			s_select_bypass_1 <=i_data(33);

			if(i_SC_disable_dec='1') then s_select_bypass_1 <= '1'; end if;
			s_select_bypass<= s_select_bypass_1;
		end if;
	end if;
end process;

--output assignment: replace TCC and ECC by their decoded counterparts when not bypassing
o_data<=s_data_bypass	when s_select_bypass='1' else
	s_data_bypass(33 downto 22) & s_data_dec(14 downto 0) & s_data_bypass(6 downto 0);
end architecture;

