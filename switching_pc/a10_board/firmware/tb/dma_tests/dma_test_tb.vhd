library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


--  A testbench has no ports.
entity dma_test_tb is
end dma_test_tb;

architecture behav of dma_test_tb is
  --  Declaration of the component that will be instantiated.

	component dma_counter is
		port(
			i_clk 			: 	in STD_LOGIC;
			i_reset_n		:	in std_logic;
			i_enable		:	in std_logic;
			i_dma_wen_reg	:	in std_logic;
			i_fraccount     :	in std_logic_vector(7 downto 0);
			i_halffull_mode	:	in std_logic;
			i_dma_halffull	:	in std_logic;
			o_dma_end_event		:	out std_logic;
			o_dma_wen		:	out std_logic;
			o_cnt 			: 	out std_logic_vector (95 downto 0)--;
			);
	end component dma_counter;

    component ip_ram is
        port(
            clock           : IN STD_LOGIC  := '1';
            data            : IN STD_LOGIC_VECTOR (95 DOWNTO 0);
            rdaddress       : IN STD_LOGIC_VECTOR (7 DOWNTO 0);
            wraddress       : IN STD_LOGIC_VECTOR (7 DOWNTO 0);
            wren            : IN STD_LOGIC  := '0';
            q               : OUT STD_LOGIC_VECTOR (95 DOWNTO 0)
        );
    end component ip_ram;

	signal clk 			: std_logic; -- 250 MHz
  	signal reset_n 		: std_logic := '1';
  	signal enable 		: std_logic := '0';
  	signal fraccount 	: std_logic_vector(7 downto 0);
  	signal dma_wen 		: std_logic;
  	signal dma_wen_reg  : std_logic;
  	signal halffull_mode  : std_logic := '0';
	signal dma_halffull  : std_logic := '0'; 
  	signal cnt 			: std_logic_vector(95 downto 0);
  	signal dma_end_event : std_logic;
    
    signal radd : std_logic_vector(7 downto 0);
    signal wadd : std_logic_vector(7 downto 0);
    signal wait_cnt : std_logic := '0';
    signal q          : std_logic_vector(95 downto 0);

  	constant ckTime		: time := 10 ns;

begin

	-- generate the clock
	ckProc: process
	begin
	   clk <= '0';
	   wait for ckTime/2;
	   clk <= '1';
	   wait for ckTime/2;
	end process;

	inita : process
	begin
	   reset_n	 <= '0';
	   wait for 8 ns;
	   reset_n	 <= '1';
	   wait for 20 ns;
	   enable    <= '1';
	   wait for 100 ns;
	   enable    <= '0';
	   wait for 50 ns;
	   enable    <= '1';

	   wait for 100 ns;
	   halffull_mode <= '1';
	   

	   wait;
	end process inita;

	fraccount <= x"FF";
	dma_wen_reg <= '1';

	e_counter : component dma_counter
    port map (
		i_clk			=> clk,
		i_reset_n   	=> reset_n,
		i_enable    	=> enable,
		i_dma_wen_reg 	=> dma_wen_reg,
		i_fraccount 	=> fraccount,
		i_halffull_mode => halffull_mode,
		i_dma_halffull 	=> dma_halffull,
		o_dma_end_event => dma_end_event,
		o_dma_wen   	=> dma_wen,
		o_cnt     		=> cnt--,
	);

    addr : process(clk, reset_n)
        variable diff : std_logic_vector(7 downto 0);
    begin
        if(reset_n = '0') then
            radd <= (others => '0');
            wadd <= (others => '0');
            wait_cnt <= '0';
            dma_halffull <= '0';
        elsif (rising_edge(clk)) then
            wait_cnt <= not wait_cnt;

            if(dma_wen = '1') then
                wadd <= wadd + '1';
            end if;

            if(wait_cnt = '1') then
                radd <= radd + '1';
            end if;

            if(wadd >= radd) then
                diff := (wadd - radd);
            else
                diff := (radd - wadd);
            end if;
            dma_halffull <= diff(7);
        end if;
    end process addr;

    e_ram : component ip_ram
    port map (
            clock       => clk,
            data        => cnt, 
            rdaddress   => radd,
            wraddress   => wadd,
            wren        => dma_wen,
            q           => q--,
        );

end behav;
