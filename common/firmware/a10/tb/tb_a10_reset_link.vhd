library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity tb_a10_reset_link is 
end entity tb_a10_reset_link;


architecture TB of tb_a10_reset_link is

    constant CLK_MHZ : real := 10000.0; -- MHz
    
    signal clk : std_logic := '0';
    signal reset_n : std_logic;

    signal rregs : work.util.slv32_array_t(63 downto 0);
    signal wregs : work.util.slv32_array_t(63 downto 0);
    
    signal reset_state : std_logic_vector(3 downto 0);

    signal delay : std_logic_vector(1 downto 0);

begin

    clk <= not clk  after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1'     after (1.0 us / CLK_MHZ);
	
    e_mapping : entity work.a10_reset_link
    generic map (
        g_XCVR2_CHANNELS    => 4--,
    )
    port map (
	    o_xcvr_tx_data      => open,
		o_xcvr_tx_datak     => open,
		
		i_reset_run_number  => wregs(RESET_LINK_RUN_NUMBER_REGISTER_W),
		i_reset_ctl         => wregs(RESET_LINK_CTL_REGISTER_W),
		i_clk               => clk,
		
		o_state_out			=> rregs(RESET_LINK_STATUS_REGISTER_R),

		i_reset_n           => reset_n--,
    );
    
    memory : process(reset_n, clk)
	begin
	if(reset_n = '0')then
		wregs(RESET_LINK_RUN_NUMBER_REGISTER_W) <= (others => '0');
		wregs(RESET_LINK_CTL_REGISTER_W) <= (others => '0');
		reset_state <= (others => '0');
        delay <= (others => '0');
		-- 
	elsif(rising_edge(clk))then
        
        delay <= delay + '1';

        if ( delay = "00" ) then

        case reset_state is
            
            when "0000" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_RUN_PREPARE_BIT) <= '1';
                wregs(RESET_LINK_RUN_NUMBER_REGISTER_W) <= x"AAAAAAAA";
                reset_state <= "0001";
                
            when "0001" =>
                reset_state <= "0010";
            
            when "0010" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_RUN_PREPARE_BIT) <= '0';
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_ABORT_RUN_BIT) <= '1';
                reset_state <= "0011";

            when "0011" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_ABORT_RUN_BIT) <= '0';
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_RUN_PREPARE_BIT) <= '1';
                wregs(RESET_LINK_RUN_NUMBER_REGISTER_W) <= x"BBBBBBBB";
                reset_state <= "0100";
            
            when "0100" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_RUN_PREPARE_BIT) <= '0';
                reset_state <= "0101";
            
            when "0101" =>
                reset_state <= "0110";
                
            when "0110" =>
                reset_state <= "0111";
            
            when "0111" =>
                reset_state <= "1000";
                
            when "1000" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_SYNC_BIT) <= '1';
                reset_state <= "1001";
                
            when "1001" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_SYNC_BIT) <= '0';
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_START_RUN_BIT) <= '1';
                reset_state <= "1010";
                
            when "1010" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_START_RUN_BIT) <= '0';
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_END_RUN_BIT) <= '1';
                reset_state <= "1011";
                
            when "1011" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_END_RUN_BIT) <= '0';
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_SYNC_BIT) <= '1';
                reset_state <= "1100";
                
            when "1100" =>
                wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_FEB_RANGE) <= wregs(RESET_LINK_CTL_REGISTER_W)(RESET_LINK_FEB_RANGE) + '1';
                reset_state <= "0000";
                
            when others =>
                reset_state <= "0000";
                
        end case;
        end if;
	end if;
	end process memory;
    
end TB;


