-- module to test the mutrig data stream. If configured accordingly, the asic produces a datastream of frames where the events are prbs48-words.
--These "events" can be checked against the expected value using this module.

-- check if the PRBS-48 pattern in the data stream is correct or not
--
-- provided previous PRBS-48 pattern and check if the current PRBS pattern is correct
--
-- the check is reset at every frame and the output is error count for each frame


Library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;


entity prbs48_checker is
	port(
		i_clk		: in std_logic;
		i_rst		: in std_logic; -- reset the checker at every frame

		i_new_word	: in std_logic;
		i_prbs_word	: in std_logic_vector(47 downto 0);

		o_err_cnt	: out std_logic_vector(7 downto 0)
	);
end entity;

architecture checker of prbs48_checker is
	signal s_pre_prbs	: std_logic_vector(47 downto 0); -- previous prbs patten
	signal s_exp_prbs	: std_logic_vector(47 downto 0); -- expected prbs patten

	signal s_first_prbs	: std_logic;

	signal s_err_cnt	: unsigned(7 downto 0);
begin
	o_err_cnt <= std_logic_vector(s_err_cnt);

	-- calculate the expected prbs for checking
	s_exp_prbs <= s_pre_prbs(s_pre_prbs'high-1 downto 0) & (s_pre_prbs(47) xor s_pre_prbs(42));

	-- checking the prbs
	proc_check : process(i_clk)
	begin
		if rising_edge(i_clk) then
			if(i_rst = '1') then
				s_err_cnt <= (others => '0');
				s_first_prbs	<= '1';
			else
				if(i_new_word = '1') then

					if(s_first_prbs = '1' ) then -- if it's the first prbs word in the frame, reset the flag
						s_first_prbs <= '0';
					else
						-- if it's not the first prbs in the frame, check the prbs word
						if (i_prbs_word /= s_exp_prbs) then
							s_err_cnt <= s_err_cnt + 1;
						end if;
					end if;

					-- latch the new prbs word for the next checking
					s_pre_prbs <= i_prbs_word;
				end if;
			end if;
		end if;
	end process;
end architecture;
