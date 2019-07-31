library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 

--  A testbench has no ports.
entity readout_tb is
end readout_tb;

architecture behav of readout_tb is
  --  Declaration of the component that will be instantiated.
	component data_generator_a10_tb is
		port(
                clk:                 	in std_logic;
                reset:               	in std_logic;
                enable_pix:          	in std_logic;
                random_seed:				in std_logic_vector (15 downto 0);
                start_global_time:		in std_logic_vector(47 downto 0);
                data_pix_generated:  	out std_logic_vector(31 downto 0);
                datak_pix_generated:  	out std_logic_vector(3 downto 0);
                data_pix_ready:      	out std_logic;
                slow_down:					in std_logic_vector (31 downto 0);
                state_out:  	out std_logic_vector(3 downto 0)
			);		
	end component data_generator_a10_tb;
    
    component event_counter is
        port (
         clk:               in std_logic;
         reset_n:           in std_logic;
         rx_data:           in std_logic_vector (31 downto 0);
         rx_datak:          in std_logic_vector (3 downto 0);
         dma_wen_reg:       in std_logic;
         event_length:      out std_logic_vector (7 downto 0);
         dma_data_wren:     out std_logic;
         dmamem_endofevent: out std_logic; 
         dma_data:          out std_logic_vector (31 downto 0);
         state_out:         out std_logic_vector(3 downto 0)
           );
    end component event_counter;


  --  Specifies which entity is bound with the component.
  		
  		signal clk : std_logic;
  		signal reset_n : std_logic := '1';
  		signal reset : std_logic;
  		signal enable_pix : std_logic;
  		signal slow_down : std_logic_vector(31 downto 0);
  		signal data_pix_generated : std_logic_vector(31 downto 0);
      signal datak_pix_generated : std_logic_vector(3 downto 0);
      signal data_pix_ready : std_logic;
      signal dmamem_endofevent : std_logic;
      signal state_out_datagen : std_logic_vector(3 downto 0);
      signal state_out_eventcounter : std_logic_vector(3 downto 0);
      signal event_length : std_logic_vector(7 downto 0);
      signal dma_data_wren : std_logic;
      signal dma_data : std_logic_vector(31 downto 0);
  		  		
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.
  
  reset <= not reset_n;
  enable_pix <= '1';
  slow_down <= x"00000000";--(others => '0');
  
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
   
   wait;
end process inita;
 
e_data_gen : component data_generator_a10_tb
	port map (
		clk 				     => clk,
		reset				     => reset,
		enable_pix	        => enable_pix,
		random_seed 		  => (others => '1'),
		start_global_time	  => (others => '0'),
		data_pix_generated  => data_pix_generated,
		datak_pix_generated => datak_pix_generated,
		data_pix_ready		  => data_pix_ready,
		slow_down			  => slow_down,
		state_out			  => state_out_datagen--,
);

e_event_counter : component event_counter
   port map(
   		clk                => clk,
         reset_n            => reset_n,
         rx_data            => data_pix_generated,
         rx_datak           => datak_pix_generated,
         dma_wen_reg        => '1',
         event_length       => event_length,
         dma_data_wren      => dma_data_wren,
         dmamem_endofevent  => dmamem_endofevent,
         dma_data           => dma_data,
         state_out          => state_out_eventcounter--,
   );

end behav;
