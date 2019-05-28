--File clock_recovery.vhd
--Recovering a 8b10 encoded data stream according to Xilinx application note 224
--
--Author: Tobias Harion
--Date: 10.10.2013



Library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.STD_LOGIC_ARITH.all;
use IEEE.STD_LOGIC_UNSIGNED.all;

entity clock_recovery is
	port(
			i_serial_data : in std_logic;

			i_rst : in std_logic;			--THE RESET FOR CLEARING THE MUX LOCK
			i_clk0 : in std_logic;		--TWO PHASESHIFTED CLOCKS USED TO SAMPLE THE DATA IN 4 TIME DOMAINS
			i_clk90 : in std_logic;

			o_syn_data : out std_logic 	--THE DATA SAMPLED AT THE INPUT BY THE CORRECT CLOCK EDGE SYNCHRONOUS TO I_CLK0
	    );
end clock_recovery;


architecture a of clock_recovery is


	--THE DATA SAMPLED IN THE INDIVIDUAL TIME DOMAINS
	signal s_data0 : std_logic_vector(3 downto 0);
	signal s_data90 : std_logic_vector(3 downto 0);
	signal s_data180 : std_logic_vector(3 downto 0);
	signal s_data270 : std_logic_vector(3 downto 0);

	signal use_0 : std_logic_vector(3 downto 0);
	signal use_90 : std_logic_vector(3 downto 0);
	signal use_180 : std_logic_vector(3 downto 0);
	signal use_270 : std_logic_vector(3 downto 0);

	--THE SIGNALS INDICATING BETWEEN WHICH DOMAINS A TRANSITION OCCURED
	signal s_transitionsP, s_transitionsN : std_logic_vector(3 downto 0);
	signal s_mux_value : std_logic_vector(3 downto 0);

	signal lock_mux : std_logic;


	attribute keep : string;
	attribute keep of s_data0 : signal is "true";
	attribute keep of s_data90 : signal is "true";
	attribute keep of s_data180 : signal is "true";
	attribute keep of s_data270 : signal is "true";


begin


		--======SAMPLING TIMEDOMAIN 0=====--	{{{

	process(i_clk0)
	begin
		if rising_edge(i_clk0) then
			s_data0 <= s_data0(2 downto 0) & i_serial_data;
		end if;
	end process;

	--}}}

		--======SAMPLING TIMEDOMAIN 1=====-- {{{
	process(i_clk0, i_clk90)
	begin

		if rising_edge(i_clk90) then
			s_data90(0) <= i_serial_data;
		end if;

		if rising_edge(i_clk0) then
			s_data90(3 downto 1) <= s_data90(2 downto 0);
		end if;
	end process;--}}}

		--======SAMPLING TIMEDOMAIN 2=====-- {{{
	process(i_clk0, i_clk90)
	begin
		if falling_edge(i_clk0) then
			s_data180(0) <= i_serial_data;
		end if;

		if rising_edge(i_clk90) then
			s_data180(1) <= s_data180(0);
		end if;

		if rising_edge(i_clk0) then
			s_data180(3 downto 2) <= s_data180(2 downto 1);
		end if;
	end process;--}}}

		--======SAMPLING TIMEDOMAIN 3=====-- {{{
	process(i_clk0, i_clk90)
	begin
		if falling_edge(i_clk90) then
			s_data270(0) <= i_serial_data;
		end if;

		if falling_edge(i_clk0) then
			s_data270(1) <= s_data270(0);
		end if;

		if rising_edge(i_clk90) then
			s_data270(2) <= s_data270(1);
		end if;

		if rising_edge(i_clk0) then
			s_data270(3) <= s_data270(2);
		end if;
	end process;--}}}


	--=====TRANSITION FINDING=====--
	p_findtrans: process(i_clk0)
	begin
		if rising_edge(i_clk0) then
			s_transitionsP(0) <= (s_data0(3) xor s_data0(2)) and not s_data0(2);		--TRANSISION IN TIMEDOMAIN 0 -> select domain 2 for data sampling
			s_transitionsP(1) <= (s_data90(3) xor s_data90(2)) and not s_data90(2);		--TRANSISION IN TIMEDOMAIN 1 -> select domain 3 for data sampling
			s_transitionsP(2) <= (s_data180(3) xor s_data180(2)) and not s_data180(2);	--TRANSISION IN TIMEDOMAIN 2 -> select domain 0 for data sampling
			s_transitionsP(3) <= (s_data270(3) xor s_data270(2)) and not s_data270(2);	--TRANSISION IN TIMEDOMAIN 3 -> select domain 1 for data sampling

			s_transitionsN(0) <= (s_data0(3) xor s_data0(2)) and s_data0(2);		--TRANSISION IN TIMEDOMAIN 0 -> select domain 2 for data sampling
			s_transitionsN(1) <= (s_data90(3) xor s_data90(2)) and s_data90(2);		--TRANSISION IN TIMEDOMAIN 1 -> select domain 3 for data sampling
			s_transitionsN(2) <= (s_data180(3) xor s_data180(2)) and s_data180(2);	--TRANSISION IN TIMEDOMAIN 2 -> select domain 0 for data sampling
			s_transitionsN(3) <= (s_data270(3) xor s_data270(2)) and s_data270(2);	--TRANSISION IN TIMEDOMAIN 3 -> select domain 1 for data sampling
		end if;
	end process;


	--=====STORE NEW MUX SELECT VALUE=====--
	p_storemux: process(i_clk0)
	begin
		if rising_edge(i_clk0) then

			--RESET THE SHIFT REGISTERS
			if i_rst = '1' then
				use_0 <= (others => '0');
				use_90 <= (others => '0');
				use_180 <= (others => '0');
				use_270 <= (others => '0');
			end if;

			if s_transitionsP = "1111" or s_transitionsN="1111" then
				use_180 <= '1' & use_180(3 downto 1);
			else
				use_180 <= '0' & use_180(3 downto 1);
			end if;

			if s_transitionsP = "0001" or s_transitionsN="0001" then
				use_270 <= '1' & use_270(3 downto 1);
			else
				use_270 <= '0' & use_270(3 downto 1);
			end if;

			if s_transitionsP = "0011" or s_transitionsN="0011" then
				use_0 <= '1' & use_0(3 downto 1);
			else
				use_0 <= '0' & use_0(3 downto 1);
			end if;

			if s_transitionsP = "0111" or s_transitionsN="0111" then
				use_90 <= '1' & use_90(3 downto 1);
			else
				use_90 <= '0' & use_90(3 downto 1);
			end if;

		end if;
	end process;



	process(i_clk0)
	begin
		if rising_edge(i_clk0) then
			if i_rst = '1' then
				lock_mux <= '0';
				s_mux_value <= (others => '0');

			elsif lock_mux = '0' then
				if use_0 = "1111" then
					lock_mux <= '1';
					s_mux_value <= "0100";
				end if;

				if use_90 = "1111" then
					lock_mux <= '1';
					s_mux_value <= "1000";
				end if;

				if use_180 = "1111" then
					lock_mux <= '1';
					s_mux_value <= "0001";
				end if;

				if use_270 = "1111" then
					lock_mux <= '1';
					s_mux_value <= "0010";
				end if;

			end if;
		end if;
	end process;

	--select the correct phase which will be passed on to the deserializer
	with s_mux_value select o_syn_data <= s_data0(3) when "0100",
										  s_data90(3) when "1000",
										  s_data180(3) when "0001",
										  s_data270(3) when "0010",
										  s_data0(3) when others;

end;
