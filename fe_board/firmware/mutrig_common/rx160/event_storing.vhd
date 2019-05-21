--STIC EVENT STORING
--
-- STORE THE RECEIVED EVENTS IN A FIFO


LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
--USE IEEE.std_logic_arith.all;
use IEEE.numeric_std.all;



entity event_storing is --{{{
	port (
			--SIGNALS FROM THE FRAME DECOMPOSER
			i_sys_clk : in std_logic;
			i_rst : std_logic;
			i_event_data : in std_logic_vector(55 downto 0);
			i_event_ready : in std_logic;
			--SIGNAL FROM THE MICROPROCESSOR
			i_rd_en : in std_logic;
			i_rd_clk : in std_logic;
			o_fifo_dout : out std_logic_vector(55 downto 0);
			o_fifo_empty : out std_logic;
			o_fifo_count : out std_logic_vector(12 downto 0);
			o_fifo_full : out std_logic;
			i_debug_events : in std_logic_vector(8 downto 0)
		 );
end entity event_storing; --}}}




architecture fast_clock of event_storing is --{{{1

--SIGNAL AND COMPONENT DECLARATION{{{2


	signal write_enable : std_logic;
	
	signal fifo_dout  : std_logic_vector(55 downto 0);
	signal fifo_empty : std_logic;
	signal fifo_count : std_logic_vector(12 downto 0);
	signal fifo_full  : std_logic;
	signal counter : std_logic_vector(55 downto 0);


	--VERY SMALL STATE MACHINE TO STORE AN EVENT ONLY ONCE FOR EACH EVENT READY
	type store_state is (S_IDLE, S_WRITE, S_RETURN);
	signal present_state, next_state : store_state;

	component generic_dp_fifo is	--{{{
		generic(
				C_DATA_WIDTH : integer := 32;
				C_ADDR_WIDTH : integer := 6;
				C_NUM_WORDS : integer := 64
			);
		port(
				i_rst : in std_logic;		--RESET SIGNAL
				clka : in std_logic;	--WRITE CLOCK SIGNAL
				clkb : in std_logic;	--READ CLOCK SIGNAL
				din : in std_logic_vector(C_DATA_WIDTH-1 downto 0);	--INPUT DATA
				we_a : in std_logic;		--WRITE ENABLE
				re_b : in std_logic;		--READ ENABLE
				dout : out std_logic_vector(C_DATA_WIDTH-1 downto 0);	--OUTPUT DATA
				count : out std_logic_vector(C_ADDR_WIDTH-1 downto 0);	--COUNT OF WORDS CURRENTLY IN THE BUFFER
				full : out std_logic;		--FIFO FULL SIGNAL
				empty : out std_logic		--FIFO EMPTY SIGNAL
			);
	end component; --}}}

--	COMPONENT fifo_56bit --{{{3
--	  PORT (
--		rst : IN STD_LOGIC;
--		wr_clk : IN STD_LOGIC;
--		rd_clk : IN STD_LOGIC;
--		din : IN STD_LOGIC_VECTOR(55 DOWNTO 0);
--		wr_en : IN STD_LOGIC;
--		rd_en : IN STD_LOGIC;
--		dout : OUT STD_LOGIC_VECTOR(55 DOWNTO 0);
--		full : OUT STD_LOGIC;
--		empty : OUT STD_LOGIC
--	  );
--	END COMPONENT; --}}}

--}}}

begin

	o_fifo_dout <= fifo_dout   when (i_debug_events = "000000000") else x"caffee" & counter(31 downto 0);
	o_fifo_empty <= fifo_empty when (i_debug_events = "000000000") else '0';
	o_fifo_count <= fifo_count when (i_debug_events = "000000000") else "0000" & i_debug_events;
	o_fifo_full <= fifo_full   when (i_debug_events = "000000000") else '0';

	fsm_storing_comb : process (present_state,i_event_ready) --CALCULATING THE NEXT STATES AND OUTPUT FUNCTION{{{2
	begin
		write_enable <= '0';
		next_state <= present_state;
		case present_state is
			when S_IDLE =>
				if i_event_ready = '1' then
					next_state <= S_WRITE;
				end if;

			when S_WRITE =>
				write_enable <= '1';
				next_state <= S_RETURN;

			when S_RETURN => 
				write_enable <= '0';
				if i_event_ready = '0' then 
					next_state <= S_IDLE;
				else
					next_state <= S_RETURN;
				end if;

			when others => 
				next_state <= S_IDLE;
				write_enable <= '0';
		end case;
	end process fsm_storing_comb; --}}}

   fsm_counter : process (i_rd_en)
       begin
           if rising_edge(i_rd_en) then
				counter <= std_logic_vector(unsigned(counter)+1);
           end if;
   end process fsm_counter;


	fsm_storing_sync : process(i_sys_clk) --ASSIGN THE NEXT STATE OR RESET THE FSM{{{2
	begin
		if rising_edge(i_sys_clk) then
			if i_rst = '1' then
				present_state <= S_IDLE;
			else
				present_state <= next_state;
			end if;
		end if;
	end process fsm_storing_sync; --}}}

ev_buffer : generic_dp_fifo --{{{2
	generic map(
				C_DATA_WIDTH => 56,
				C_ADDR_WIDTH => 13,
				C_NUM_WORDS => 8192
			   )
	PORT MAP (
				i_rst => i_rst,
				clka => i_sys_clk,
				clkb => i_rd_clk,
				din => i_event_data,
				we_a => write_enable,
				re_b => i_rd_en,
				dout => fifo_dout,
				count => fifo_count,
				full => fifo_full,
				empty => fifo_empty 
		  ); --}}}

  
  
--  ev_buffer : fifo_56bit --{{{2
--  PORT MAP (
--    rst => i_rst,
--    wr_clk => i_sys_clk,
--    rd_clk => i_rd_clk,
--    din => i_event_data,
--    wr_en => write_enable,
--    rd_en => i_rd_en,
--    dout => o_fifo_dout,
--    full => o_fifo_full,
--    empty => o_fifo_empty 
--  ); --}}}


end architecture fast_clock;--}}}
