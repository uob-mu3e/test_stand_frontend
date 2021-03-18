-----------------------------------
--
-- Removing SC information from hit data stream for MP10
-- Sebastian Dittmeier, June 2020
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;
	

entity mp_sc_removal is 
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		is_MP10				: in std_logic;
		sc_active			: in std_logic;
		new_block			: in std_logic;
		hit_in				: in std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_in			: in std_logic;
		coarsecounters_ena: in std_logic;	-- has to be '0' for hits!
		hit_out				: out std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_out			: out std_logic
		);
end mp_sc_removal;


architecture rtl of mp_sc_removal is

-- it may be only the last two hits of a readout block, that could be SC
-- so we simply store the last two hits of a readout cycle
-- we check the following signals:
-- is_MP10: only if = '1' then we take action, otherwise bypass
-- sc_active: only if = '1' then we take action, otherwise bypass
-- new_block: indicates, that a readout cylce has passed, then we delete last two hits
-- we do that by erasing the hin_ena flag
-- we can use the link_id_flag for that
	signal hit_reg_1 			: std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
	signal hit_reg_2 			: std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
	signal hit_reg_out		: std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
	signal hit_ena_reg_1		: std_logic;
	signal hit_ena_reg_2		: std_logic;
	signal hit_ena_reg_out 	: std_logic;


begin

	-- we have to blindly forward coarsecounter data on hit lines
	hit_out 		<= 	hit_in				when coarsecounters_ena = '1' else
							hit_reg_out 		when (is_MP10 = '1' and sc_active = '1') else
							hit_in;
	
	-- we have to blindly forward coarsecounters_ena signal
	hit_ena_out <= 	hit_ena_in			when coarsecounters_ena = '1' else	
							hit_ena_reg_out 	when (is_MP10 = '1' and sc_active = '1') else
							hit_ena_in;

	process(reset_n, clk)
	begin
	if(reset_n = '0')then
		hit_reg_1 		<= (others => '0');
		hit_reg_2 		<= (others => '0');
		hit_reg_out		<= (others => '0');
		hit_ena_reg_1 	<= '0';
		hit_ena_reg_2 	<= '0';
		hit_ena_reg_out	<= '0';
	elsif(rising_edge(clk))then
		hit_ena_reg_out	<= '0';	-- defaults to low
		if(new_block = '1')then
			hit_ena_reg_1 <= '0';	-- we delete the stored flags
			hit_ena_reg_2 <= '0';	-- we delete the stored flags
		elsif(hit_ena_in = '1' and coarsecounters_ena = '0')then	-- here we deal with hits
			hit_reg_1		<= hit_in;
			hit_reg_2		<= hit_reg_1;
			hit_reg_out		<= hit_reg_2;		-- SC data would actually pop out as hits, but the enable signals are deleted
				
			hit_ena_reg_1	<= hit_ena_in;
			hit_ena_reg_2	<= hit_ena_reg_1;
			hit_ena_reg_out<= hit_ena_reg_2;	
		end if;
	end if;
	end process;

end rtl;
