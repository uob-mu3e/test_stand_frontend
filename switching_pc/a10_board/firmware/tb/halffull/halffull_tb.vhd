library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


--  A testbench has no ports.
entity halffull_tb is
end entity;

architecture behav of halffull_tb is
  --  Declaration of the component that will be instantiated.

    component dma_counter is
        port(
            i_clk           :   in STD_LOGIC;
            i_reset_n       :   in std_logic;
            i_enable        :   in std_logic;
            i_dma_wen_reg   :   in std_logic;
            i_fraccount     :   in std_logic_vector(7 downto 0);
            i_halffull_mode :   in std_logic;
            i_dma_halffull  :   in std_logic;
            o_dma_end_event     :   out std_logic;
            o_dma_wen       :   out std_logic;
            o_cnt           :   out std_logic_vector (95 downto 0)--;
            );
    end component dma_counter;

	signal clk 			: std_logic := '0';
    signal clk2          : std_logic := '0';
  	signal reset_n 		: std_logic := '1';

    signal enable      : std_logic;
    signal fraccount    : std_logic_vector(7 downto 0);
    signal dma_wen      : std_logic;
    signal dma_wen_reg  : std_logic;
    signal halffull_mode  : std_logic := '0';
    signal dma_halffull  : std_logic := '0'; 
    signal cnt          : std_logic_vector(95 downto 0);
    signal dma_end_event : std_logic;

    signal memhalffull : std_logic;
  	signal wadd : std_logic_vector(4 downto 0);
    signal radd : std_logic_vector(4 downto 0);

  	constant ckTime		: time := 10 ns;
    constant ckTime2     : time := 20 ns;

begin

	-- generate the clock
	ckProc: process
	begin
	   clk <= '0';
	   wait for ckTime;
	   clk <= '1';
	   wait for ckTime;
	end process;

    ckProc2: process
    begin
       clk2 <= '0';
       wait for ckTime2;
       clk2 <= '1';
       wait for ckTime2;
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

    -- write add
    process(clk, reset_n)
    begin
        if(reset_n = '0') then
            wadd <= (others => '0');
        elsif(clk'event and clk = '1') then
            if(memhalffull = '0') then
                wadd <= wadd + '1';     
            end if;
        end if;
    end process;

    -- read add
    process(clk2, reset_n)
    begin
        if(reset_n = '0') then
            radd <= (others => '0');
        elsif(clk2'event and clk2 = '1') then
            radd <= radd + '1';     
        end if;
    end process;

    -- rhalffull
    process(clk, reset_n)
    variable diff : std_logic_vector(4 downto 0);
    begin
        if(reset_n = '0') then
            memhalffull <= '0';
        elsif(clk2'event and clk2 = '1') then
            if(wadd >= radd) then
                diff := (wadd - radd);
            else
                diff := (radd - wadd);
            end if;
                memhalffull <= diff(4); 
            end if;
    end process;

    fraccount <= x"FF";
    dma_wen_reg <= '1';

    e_counter : component dma_counter
    port map (
        i_clk           => clk,
        i_reset_n       => reset_n,
        i_enable        => enable,
        i_dma_wen_reg   => dma_wen_reg,
        i_fraccount     => fraccount,
        i_halffull_mode => halffull_mode,
        i_dma_halffull  => memhalffull,
        o_dma_end_event => dma_end_event,
        o_dma_wen       => dma_wen,
        o_cnt           => cnt--,
    );

end architecture;
