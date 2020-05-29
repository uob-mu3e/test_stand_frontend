-- ipbus_i2c_master
--
-- Wrapper for opencores i2c wishbone slave
--
-- Dave Newbold, Jan 2012

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.ipbus.all;

entity ipbus_i2c_master is
	generic(addr_width: natural := 0);
	port(
		clk: in std_logic;
		rst: in std_logic;
		ipbus_in: in ipb_wbus;
		ipbus_out: out ipb_rbus;
		ipbus_in_fast: in ipb_wbus;
        ipbus_out_fast: out ipb_rbus;
 		ipbus_in_mem: in ipb_wbus;
        ipbus_out_mem: out ipb_rbus;       
		scl_o: out std_logic;
		scl_i: in std_logic;
		sda_o: out std_logic;
		sda_i: in std_logic
	);
	
end ipbus_i2c_master;

architecture rtl of ipbus_i2c_master is

	signal stb, ack, ack_last, sda_enb: std_logic;
	
	signal addr:       std_logic_vector(2 downto 0);
	signal data_in:    std_logic_vector(7 downto 0);
	signal data_out:   std_logic_vector(7 downto 0);
	signal we:         std_logic;
	
	type state_type is (idle, simple, fast, mem);
	signal state : state_type;

begin

    process(clk, rst)
    begin
    if(rst = '1') then
        state <= idle;
        stb  <= '0';
        we   <= '0';
        ack_last    <= '0';
    elsif(rising_edge(clk)) then
        ack_last <= ack;
        stb  <= '0';
        case state is
            when idle =>
                if(ipbus_in.ipb_strobe = '1')then
                    state   <=  simple;
                    addr    <=  ipbus_in.ipb_addr(2 downto 0);
                    data_in <=  ipbus_in.ipb_wdata(7 downto 0);
                    we      <=  ipbus_in.ipb_write;
                    stb     <= '1';
                elsif(ipbus_in_fast.ipb_strobe = '1') then
                    state   <= fast;
                elsif(ipbus_in_mem.ipb_strobe = '1') then
                    state   <= mem;
                end if;
            when simple =>
                ipbus_out.ipb_rdata(7 downto 0) <= data_out;
                ipbus_out.ipb_rdata(31 downto 8) <= (others => '0');
                ipbus_out.ipb_err <= '0';
                ipbus_out.ipb_ack <= ack;
                if(ack = '1') then
                    state <= idle;
                    we    <= '0';
                 end if;
            when fast =>
                state <= idle;    
            when mem =>
                state <= idle;
        end case;
    end if;
    end process;


	--stb <= ipbus_in.ipb_strobe and not ack;

	i2c: entity work.i2c_master_top port map(
		wb_clk_i => clk,
		wb_rst_i => rst,
		arst_i => '1',
		wb_adr_i => addr,
		wb_dat_i => data_in,
		wb_dat_o => data_out,
		wb_we_i => we,
		wb_stb_i => stb,
		wb_cyc_i => '1',
		wb_ack_o => ack,
		scl_pad_i => scl_i,
		scl_padoen_o => scl_o,
		sda_pad_i => sda_i,
		sda_padoen_o => sda_o
	);
		
end rtl;
