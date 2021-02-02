library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.dataflow_components.all;


entity merging_demerging_tb is 
end entity merging_demerging_tb;


architecture TB of merging_demerging_tb is

    signal reset_n		: std_logic;
    signal reset        : std_logic;
    signal clk		    : std_logic;

    constant NLINKS     : positive := 8;

    signal r_fifo_data, w_fifo_data  : std_logic_vector(NLINKS * 38 - 1 downto 0);
    signal w_fifo_en, r_fifo_en, fifo_empty, fifo_full : std_logic;

    type sim_merger_state_type is (pre, t1, t2, sh, data, tr);
    signal event_counter_state : sim_merger_state_type;

    signal data_counter : std_logic_vector(31 downto 0);

    -- clk period
    constant clk_period : time := 4 ns;


begin
	
	--clk
	process begin
		clk <= '0';
		wait for clk_period;
		clk <= '1';
		wait for clk_period;
	end process;
	
	-- reset_n
	process begin
		reset_n <= '0';
		wait for 20 ns;
		reset_n <= '1';
		wait;
	end process;

    reset <= not reset_n;

    -- write to merger
    -- TODO: here the time alignment will be paced
    process(clk, reset_n)
    begin
    if(reset_n <= '0') then
        w_fifo_data         <= (others => '0');
        data_counter        <= (others => '0');
        w_fifo_en           <= '0';
        event_counter_state <= pre;
    elsif(clk'event and clk = '1') then

        w_fifo_en   <= '0';
        w_fifo_data <= (others => '0');

        case event_counter_state is
            when pre =>
                event_counter_state <= t1;
                w_fifo_data(37 downto 32) <= pre_marker;
                w_fifo_data(7 downto 0) <= x"BC";
                w_fifo_en <= '1';
            when t1 =>
                event_counter_state <= t2;
                w_fifo_data(37 downto 32) <= ts1_marker;
                w_fifo_en <= '1';
            when t2 =>
                event_counter_state <= sh;
                w_fifo_data(37 downto 32) <= ts2_marker;
                w_fifo_en <= '1';
            when sh =>
                event_counter_state <= data;
                w_fifo_data(37 downto 32) <= sh_marker;
                w_fifo_en <= '1';
            when data =>
                if ( data_counter = x"000000FF" ) then
                    event_counter_state <= tr;    
                end if;
                data_counter <= data_counter + '1';
                w_fifo_data(37 downto 0)    <= "000000" & x"11111111";
                w_fifo_data(75 downto 38)   <= "000001" & x"22222222";
                w_fifo_data(113 downto 76)  <= "000010" & x"33333333";
                w_fifo_data(151 downto 114) <= "000011" & x"44444444";
                w_fifo_data(189 downto 152) <= "000100" & x"55555555";
                w_fifo_data(227 downto 190) <= "000101" & x"66666666";
                w_fifo_data(265 downto 228) <= "000110" & x"77777777";
                w_fifo_data(303 downto 266) <= "000111" & x"88888888";
                w_fifo_en <= '1';
            when tr =>
                event_counter_state <= pre;
                w_fifo_data(37 downto 32) <= tr_marker;
                w_fifo_en <= '1';
            when others =>
                event_counter_state <= pre;
        end case;
    end if;
    end process;

    -- TODO: here the time alignment will be paced
    e_merger_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_fifo_data,
        wrreq           => w_fifo_en,
        rdreq           => r_fifo_en,
        clock           => clk,
        q               => r_fifo_data,
        full            => fifo_full,
        empty           => fifo_empty,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => reset--,
    );

    -- data merger swb
    e_data_merger_swb : entity work.data_merger_swb
    generic map (
        NLINKS  => NLINKS,
        DT      => x"01"--,
    )
    port map (
        i_reset_n   => reset_n,
        i_clk       => clk,
        
        i_data      => r_fifo_data,
        i_empty     => fifo_empty,

        i_swb_id    => x"01",
        
        o_ren       => r_fifo_en,
        o_wen       => open,
        o_data      => open,
        o_datak     => open--,
    );


end TB;


