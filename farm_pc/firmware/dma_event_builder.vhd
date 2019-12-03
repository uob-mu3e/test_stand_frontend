library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity dma_event_builder is
	generic (
		endofevent_marker	: std_logic_vector := x"9c"
	);
    port(
		clk:        			in std_logic;
		reset_n:    			in std_logic;
		enable:	 				in std_logic;
		i_data:					in std_logic_vector(31 downto 0);
		i_datak:					in std_logic_vector(3 downto 0);
		o_endofevent:			out std_logic;
		o_eventlength:			out std_logic_vector(11 downto 0);
		o_data:					out std_logic_vector(31 downto 0);
);
end entity dma_event_builder;

architecture rtl of dma_event_builder is

		signal dma_control_wren 		: std_logic;
		signal dma_control_counter		: std_logic_vector(15 downto 0);
		signal dma_control_prev_rdreq : std_logic_vector(15 downto 0);
		type event_counter_state_type is (waiting, ending);
		signal event_counter_state : event_counter_state_type;
		signal event_tagging_state : event_counter_state_type;
		signal w_ram_en	 : std_logic;
		signal w_fifo_en	 : std_logic;
		signal w_fifo_data : std_logic_vector(11 downto 0);
		signal w_ram_data	 : std_logic_vector(31 downto 0);
		signal w_ram_add	 : std_logic_vector(11 downto 0);
		signal tag_fifo_empty : std_logic;
		signal r_fifo_data : std_logic_vector(11 downto 0);
		signal r_fifo_en : std_logic;
		signal r_ram_data : std_logic_vector(31 downto 0);
		signal r_ram_add  : std_logic_vector(11 downto 0);
		signal event_last_ram_add : std_logic_vector(11 downto 0);
		signal data_pix_generated : std_logic_vector(31 downto 0);
		signal data_pix_ready : std_logic;
		signal data_pix_generated2 : std_logic_vector(31 downto 0);
		signal data_pix_ready2 : std_logic;
		signal event_length : std_logic_vector(11 downto 0);