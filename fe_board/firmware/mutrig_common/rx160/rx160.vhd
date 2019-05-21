-- stic 3 data receiver
-- Simon Corrodi based on KIP DAQ
-- July 2017

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;

use work.rx160_components.all;

entity rx160 is
  port (
    o_byte_clk     : out std_logic;                     -- 
    o_sync_found   : out std_logic;                     -- debug

    --i_start_val   : in  std_logic_vector(3 downto 0);  -- bit slip for 10bit word alignment
    i_rst		    : in  std_logic;                     --
    i_stic_txd     : in  std_logic;                     -- serial data
    i_stic_clk     : in  std_logic;                     -- dara ref clk
    i_stic_clk90   : in  std_logic;                     -- data ref clk, 90 deg phaseshift for cdr

    o_event_data   : out std_logic_vector(55 downto 0); -- decoded event data
    o_event_ready  : out std_logic;                     -- decoded event data ready
	 o_end_of_frame : out std_logic                      -- end of lvds frame, can be used to trigger daq
  );
end entity rx160;

architecture RTL of rx160 is

  signal byte_data   : std_logic_vector(7 downto 0);
  signal kontrol_bit : std_logic;
  signal byte_clk    : std_logic;
  --signal event_ready : std_logic;
  --signal event_data  : std_logic_vector(55 downto 0);
  signal s_syn_data  : std_logic;	                  -- input data sync with ref clk


begin

o_byte_clk <= byte_clk;

  u_clock_recover: clock_recovery -- used becaus gxb only works from 800Gbs on
    port map(
      i_serial_data => i_stic_txd,
      i_rst         => i_rst,
      i_clk0        => i_stic_clk,
      i_clk90       => i_stic_clk90,
      o_syn_data    => s_syn_data
    );

  u_deser: deserializer -- deserializer and 8b/10b decoding
    port map(
      o_sync_found => o_sync_found,
      -- i_start_val  => i_start_val,
      i_rst        => i_rst,
      i_ser_clk    => i_stic_clk,
      i_serial_data => s_syn_data,
      o_byte_clock => byte_clk,
      o_byte_data  => byte_data,
      o_kontrol    => kontrol_bit
    );

  u_unframe: frame_decomp -- for STiC 3.2
    port map(
      i_k_signal     => kontrol_bit,
      i_byte_data    => byte_data,
      i_byte_clk     => byte_clk,
      i_rst          => i_rst,
      o_event_data   => o_event_data,
      o_event_ready  => o_event_ready,
		o_end_of_frame => o_end_of_frame
    );

end architecture RTL;
