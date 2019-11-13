-- PLL with register interface to reconfigure the dynamic phase settings of an ALTPLL IP
-- input: write flag - action will be taken if raised
-- input: data[0]: If set, increment phase counter. If not set, decrement the counter.
-- input: data[4..1]: Selects PLL output to be reconfigured.
-- input: data[6..5]: Sets phase & 1cc shift of line 0
-- input: data[8..7]: Sets phase & 1cc shift of line 1

Library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.STD_LOGIC_ARITH.all;
use IEEE.STD_LOGIC_UNSIGNED.all;


entity clockalign_block is
	generic(
			CLKDIV  : integer := 125--;
	);
	port(
		--SYSTEM SIGNALS
		i_clk_config : in std_logic;
		i_rst : in std_logic;

		i_flag  : in std_logic;
		i_data  : in std_logic_vector(31 downto 0);

		i_pll_clk : in std_logic;
		i_pll_arst : in std_logic;
		o_pll_clk : out std_logic_vector(3 downto 0);
		o_pll_locked : out std_logic;
		i_sig : in std_logic;
		o_sig : out std_logic_vector(3 downto 0)
	    );
end entity clockalign_block;




architecture a of clockalign_block is
	component pll_ip is
	PORT
	(
		areset		: IN STD_LOGIC  := '0';
		inclk0		: IN STD_LOGIC  := '0';
		phasecounterselect		: IN STD_LOGIC_VECTOR (3 DOWNTO 0) :=  (OTHERS => '0');
		phasestep		: IN STD_LOGIC  := '0';
		phaseupdown		: IN STD_LOGIC  := '0';
		scanclk		: IN STD_LOGIC  := '1';
		c0		: OUT STD_LOGIC ;
		c1		: OUT STD_LOGIC ;
		locked		: OUT STD_LOGIC ;
		phasedone		: OUT STD_LOGIC 
	);
	end component;

	signal s_chainclk : std_logic;
	signal s_run   : std_logic;

	signal n_PHASECOUNTERSELECT, s_PHASECOUNTERSELECT	: std_logic_vector(3 downto 0);
	signal n_PHASEUPDOWN, s_PHASEUPDOWN		: std_logic;
	signal n_PHASESTEP, s_PHASESTEP		: std_logic;
	signal s_PHASEDONE : std_logic;
	signal s_PHASEDONE_lat : std_logic;

	type fsm_states is (FS_IDLE, FS_TICK, FS_WAIT);
	signal s_state, n_state : fsm_states;

	signal s_pll_clk          : std_logic_vector(3 downto 0);
	signal s_sig_r, s_sig_r_d : std_logic_vector(3 downto 0);
	signal s_sig_f, s_sig_f_d : std_logic_vector(3 downto 0);
begin
	gen_clkdiv: if CLKDIV > 1 generate
		    e_chainclk_div : entity work.clkdiv
		    generic map ( P => CLKDIV )
		    port map ( clkout => s_chainclk, rst_n => not i_rst, clk => i_clk_config );
	end generate gen_clkdiv;

	gen_noclkdiv: if CLKDIV = 1 generate
		s_chainclk <= i_clk_config;
	end generate gen_noclkdiv;


	p_sample_start: process(i_clk_config)
	begin
		if(rising_edge(i_clk_config)) then
			if(i_rst='1') then
				s_run <= '0';
			elsif(s_state /= FS_IDLE) then
				s_run <= '0';
			elsif(i_flag='1') then
				s_run <= '1';
			end if;
		end if;
	end process;

        p_sample_PHASEDONE: process(i_clk_config, s_state, s_PHASEDONE)
	begin
		if(rising_edge(i_clk_config)) then
			if(i_rst='1') then
				s_PHASEDONE_lat <= '1';
			elsif(s_PHASEDONE='0') then
				s_PHASEDONE_lat <= '0';
			elsif(s_state /= FS_TICK) then
				s_PHASEDONE_lat <= '1';
			end if;
			
		end if;
	end process;

	fsm_comb : process (i_rst, s_state, i_data, s_run, i_flag, s_PHASEUPDOWN, s_PHASECOUNTERSELECT, s_PHASEDONE, s_PHASEDONE_lat)	--{{{
	begin
		n_state <= s_state;
		n_PHASEUPDOWN <= s_PHASEUPDOWN;
		n_PHASECOUNTERSELECT <= s_PHASECOUNTERSELECT;
		n_PHASESTEP <= '0';

		if( i_rst = '1') then 
			n_state <= FS_IDLE;
			n_PHASECOUNTERSELECT <= "1011";
			n_PHASEUPDOWN <= '0';

		else case s_state is
			when FS_IDLE =>
				if s_run = '1' then
					n_state <= FS_TICK;
					-- set configuration vector outputs to pll IP
					n_PHASECOUNTERSELECT <= i_data(4 downto 1);
					n_PHASEUPDOWN <= i_data(0);
					n_PHASESTEP <= '1';
				end if;

			when FS_TICK =>
				n_PHASESTEP <= '1';
				if(s_PHASEDONE_lat='0') then 
					n_state <= FS_WAIT;
				end if;
			when FS_WAIT =>
				n_PHASESTEP <= '1';
				if s_PHASEDONE = '1' then
					n_state <= FS_IDLE;
				end if;
			when others => NULL;
		end case;
		end if;
	end process fsm_comb;	--}}}


	fsm_syn : process (s_chainclk) --{{{
	begin
		if rising_edge(s_chainclk) then
			s_state <= n_state;
			s_PHASECOUNTERSELECT <= n_PHASECOUNTERSELECT;
			s_PHASEUPDOWN <= n_PHASEUPDOWN;
			s_PHASESTEP <= n_PHASESTEP;
		end if;
	end process fsm_syn; --}}}



	pll: pll_ip
	port map(
		areset             => i_pll_arst,                  
		inclk0             => i_pll_clk,
		phasecounterselect => s_PHASECOUNTERSELECT,
		phasestep          => s_PHASESTEP,
		phaseupdown        => s_PHASEUPDOWN,
		scanclk            => s_chainclk,
		c0                 => s_pll_clk(0),
		c1                 => s_pll_clk(1),
		--c2                 => s_pll_clk(2),
		--c3                 => s_pll_clk(3),
		locked             => o_pll_locked,
		phasedone          => s_PHASEDONE
	);
	s_pll_clk(2) <= i_pll_clk; --TODO: extend pll
	s_pll_clk(3) <= i_pll_clk; --TODO: extend pll

	o_pll_clk <= s_pll_clk;

	g_select_shift: for i in 0 to 3 generate
		signal_sync: process(s_pll_clk, i_sig)
		begin
			if rising_edge(s_pll_clk(i)) then
				s_sig_r(i) <= i_sig;
				s_sig_r_d(i) <= s_sig_r(i);
			end if;
			if falling_edge(s_pll_clk(i)) then
				s_sig_f(i) <= i_sig;
				s_sig_f_d(i) <= s_sig_f(i);
			end if;
		end process;

		with (i_data(5+i*2+1 downto 5+i*2)) select o_sig(i) <= 
			s_sig_r(i) when "00",
			s_sig_r_d(i) when "01",
			s_sig_f(i) when "10",
			s_sig_f_d(i) when "11",
			s_sig_r(i) when others;
	end generate;
end architecture;

