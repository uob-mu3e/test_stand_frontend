library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all;
use work.dataflow_components.all;


--  A testbench has no ports.
entity tree_tb is
end entity;

architecture behav of tree_tb is
  --  Specifies which entity is bound with the component.
      signal clk : std_logic;
  	  signal reset_n : std_logic := '1';
  	  signal reset : std_logic;

      signal i_rdata                  : fifo_array_38(3 downto 0);
      signal rdata                    : fifo_array_32(3 downto 0);
      signal fifo_data_0              : fifo_array_38(3 downto 0);
      signal fifo_q_0, fifo_q_0_reg   : fifo_array_76(3 downto 0);
      signal layer_0_state            : fifo_array_4(3 downto 0);
      signal layer_0_cnt              : fifo_array_32(3 downto 0);
      signal fifo_ren_0, fifo_ren_0_reg, i_mask_n, fifo_wen_0, fifo_full_0, fifo_empty_0, fifo_empty_0_reg, saw_header_0, saw_trailer_0, reset_fifo_0 : std_logic_vector(3 downto 0);
      
      signal fifo_q_1 : fifo_array_76(1 downto 0);
      signal fifo_ren_1, fifo_full_1, fifo_empty_1 : std_logic_vector(1 downto 0);
      signal o_fifo_empty_1 : std_logic_vector(0 downto 0);

      signal o_fifo_q_2 : std_logic_vector(75 downto 0);
      signal fifo_q_2 : std_logic_vector(63 downto 0);
  		  		
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.

reset <= not reset_n;
i_mask_n <= x"F";
  
  -- generate the clock
process
begin
   clk <= '0';
   wait for ckTime/2;
   clk <= '1';
   wait for ckTime/2;
end process;

process
begin
    reset_n	 <= '0';
    wait for 8 ns;
    reset_n	 <= '1';
    wait;
end process;

tree_layer_first:
FOR i in 0 to 3 GENERATE

process(clk, reset_n)
begin
if ( reset_n /= '1' ) then
  i_rdata(i) <= (others => '0');
  rdata(i) <= (others => '0');
  --
elsif rising_edge(clk) then
  if ( layer_0_state(i) /= "0001" ) then
    if ( i_rdata(i)(31 downto 28) + '1' = "1111" ) then
      i_rdata(i) <= (others => '1');
      rdata(i) <= (others => '1');
    else
      i_rdata(i)(31 downto 28) <= i_rdata(i)(31 downto 28) + '1';
      rdata(i)(31 downto 28) <= rdata(i)(31 downto 28) + '1';
    end if;
  else
    i_rdata(i) <= (others => '0');
    rdata(i) <= (others => '0');
  end if;  
end if;
end process;

    e_link_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_WADDR_WIDTH => 7,
        g_WDATA_WIDTH => 32+6,
        g_RADDR_WIDTH => 7,
        g_RDATA_WIDTH => 64+12--,
    )
    port map (
        i_we        => fifo_wen_0(i),
        i_wdata     => fifo_data_0(i),
        o_wfull     => fifo_full_0(i),
        i_wclk      => clk,

        i_rack      => fifo_ren_0(i),
        o_rdata     => fifo_q_0_reg(i),
        o_rempty    => fifo_empty_0_reg(i),
        i_rclk      => clk,

        i_reset_n   => reset_n and (not reset_fifo_0(i))--,
    );

process(clk, reset_n)
begin
if ( reset_n /= '1' ) then
    fifo_data_0(i) <= (others => '0');
    fifo_wen_0(i) <= '0';
    layer_0_state(i) <= "0000";
    reset_fifo_0(i) <= '0';
    --
elsif rising_edge(clk) then
    fifo_wen_0(i) <= '0';
    reset_fifo_0(i) <= '0';
  
    case layer_0_state(i) is
        
        when "0000" =>
            if ( i_mask_n(i) = '0' ) then
                saw_header_0(i) <= '1';
                saw_trailer_0(i) <= '1';
                layer_0_state(i) <= "0001";
                fifo_data_0(i) <= tree_padding;
                fifo_wen_0(i) <= '1';
            elsif ( fifo_full_0(i) = '1' or reset_fifo_0(i) = '1' ) then
                --
            else
                if ( fifo_full_0(i) = '0' and i_rdata(i)(37 downto 36) = "00" ) then
                    fifo_data_0(i) <= work.mudaq.link_36_to_std(i) & i_rdata(i)(35 downto 4);
                    fifo_wen_0(i) <= '1';
                    saw_header_0(i) <= '0';
                    saw_trailer_0(i) <= '0';
                -- TODO: is this fine to quite until one is written (cnt > 0)?
                else
                    layer_0_state(i) <= "0001";
                    fifo_data_0(i) <= tree_padding;
                    fifo_wen_0(i) <= '1';
                end if;
            end if;
        when "0001" =>
          fifo_data_0(i) <= tree_padding;
          fifo_wen_0(i) <= '1';
          if ( i_rdata(i)(37 downto 36) = "00" ) then
            layer_0_state(i) <= "0000";
          end if;
        when others =>
            layer_0_state(i) <= "0000";
    end case;

end if;
end process;

process(clk, reset_n)
begin
if ( reset_n /= '1' ) then
    fifo_q_0(i) <= (others => '0');
    fifo_empty_0(i) <= '0';
    --
elsif rising_edge(clk) then
    fifo_q_0(i) <= fifo_q_0_reg(i);
    fifo_empty_0(i) <= fifo_empty_0_reg(i);
end if;
end process;
END GENERATE;

layer_1 : entity work.time_merger_tree_fifo_64
generic map (  
    TREE_w => 7, TREE_r => 7,
    r_width => 64+12, w_width => 64+12,
    compare_fifos => 4, gen_fifos => 2--,
)
port map (
    i_fifo_q        => fifo_q_0,
    i_fifo_empty    => fifo_empty_0,
    i_fifo_ren      => fifo_ren_1,
    i_merge_state   => '1',
    i_mask_n        => "1111",

    o_fifo_q        => fifo_q_1,
    o_fifo_empty    => fifo_empty_1,
    o_fifo_ren      => fifo_ren_0,
    o_mask_n        => open,

    i_reset_n       => reset_n,
    i_clk           => clk--,
);

layer_2 : entity work.time_merger_tree_fifo_64
generic map (  
    TREE_w => 7, TREE_r => 7,
    r_width => 64+12, w_width => 64+12,
    compare_fifos => 2, gen_fifos => 1--,
)
port map (
    i_fifo_q        => fifo_q_1,
    i_fifo_empty    => fifo_empty_1,
    i_fifo_ren(0)      => not o_fifo_empty_1(0),
    i_merge_state   => '1',
    i_mask_n        => "11",

    o_fifo_q(0)     => o_fifo_q_2,
    o_fifo_empty(0)    => o_fifo_empty_1(0),
    o_fifo_ren      => fifo_ren_1,
    o_mask_n        => open,

    i_reset_n       => reset_n,
    i_clk           => clk--,
);

fifo_q_2(31 downto 0) <= o_fifo_q_2(31 downto 0);
fifo_q_2(63 downto 32) <= o_fifo_q_2(69 downto 38);


end architecture;
