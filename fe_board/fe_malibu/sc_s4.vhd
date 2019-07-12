----------------------------------------------------------------------------
-- Slow Control Unit for Frontend Board
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sc_s4 is
port (
    link_data_in:       in std_logic_vector(31 downto 0);
    link_data_in_k:     in std_logic_vector(3 downto 0);
    
    fifo_data_out:      out std_logic_vector(35 downto 0);
    fifo_we:            out std_logic;

    mem_data_in:        in std_logic_vector(31 downto 0);
    mem_data_out:       out std_logic_vector(31 downto 0);
    mem_addr_out:       out std_logic_vector(15 downto 0);
    mem_wren:           out std_logic;

    reset_n         : in    std_logic;
    clk             : in    std_logic
);
end entity;

architecture arch of sc_s4 is

    signal mem_data_o : std_logic_vector(31 downto 0);
    signal mem_data_i : std_logic_vector(31 downto 0);
    signal mem_addr_write_o : std_logic_vector(15 downto 0);
    signal mem_addr_read_o : std_logic_vector(15 downto 0);
    signal mem_wren_o : std_logic;

    signal start_add, end_add : unsigned(15 downto 0);

    signal idle_counter : unsigned(15 downto 0);

    signal sc_type : std_logic_vector(1 downto 0);

    type state_type is (waiting, starting, get_length, writing, reading, end_reading);
    signal state : state_type;

begin

    mem_data_out <= mem_data_o;
    mem_addr_out <= mem_addr_write_o when mem_wren_o = '1' else mem_addr_read_o;
    mem_wren     <= mem_wren_o;
    mem_data_i	 <= mem_data_in;

    process(clk, reset_n)
    begin
    if ( reset_n = '0' ) then
        state <= waiting;
        mem_data_o 		 	<= (others => '0');
        mem_addr_write_o 	<= (others => '0');
        mem_addr_read_o 	<= (others => '0');
        start_add 		  	<= (others => '0');
        end_add 		  	<= (others => '0');
        fifo_data_out		<= (others => '0');
        sc_type				<= (others => '0');
        idle_counter		<= (others => '0');
        mem_wren_o 			<= '0';
        fifo_we 			<= '0';
        --
    elsif rising_edge(clk) then
        mem_data_o 				<= (others => '0');
        --mem_addr_read_o			<= (others => '0');
        mem_addr_write_o 		<= (others => '0');
        mem_wren_o 				<= '0';
        fifo_we 				<= '0';

        if ( link_data_in(7 downto 0) = x"BC" and
             link_data_in_k = "0001" and
             link_data_in(31 downto 26) = "000111"
        ) then
            -- link preamble
            sc_type <= link_data_in(25 downto 24);
            state <= starting;
            --
        else
            if ( idle_counter = (idle_counter'range => '1') ) then
                -- timeout
                state <= waiting;
            end if;

            if ( link_data_in = x"000000BC" and link_data_in_k = "0001" ) then
                -- link idle
                idle_counter <= idle_counter + 1;
            end if;

            case state is
            when waiting => -- wait for preamble
                idle_counter <= (others => '0');
                --
            when starting => -- get start_add
                    fifo_data_out			<= sc_type & "10" & link_data_in(31 downto 0);
                    fifo_we					<= '1';
                    start_add 				<= unsigned(link_data_in(15 downto 0));
                    state 					<= get_length;
                --
            when get_length => -- get end_add and send acknowledge
                    end_add 							<= unsigned(link_data_in(15 downto 0)) + start_add;
                    fifo_data_out(35 downto 17)			<= (others => '0');
                    fifo_data_out(16) 			 		<= '1';
                    fifo_data_out(15 downto 0)	 		<= link_data_in(15 downto 0);
                    fifo_we								<= '1';
                    if (sc_type = "10") then -- read
                        state 							<= reading;
                        mem_addr_read_o 				<= std_logic_vector(start_add);
                    elsif (sc_type = "11") then -- write
                        state 							<= writing;
                    end if;
                --
            when writing =>
                    if (link_data_in = x"0000009C" and link_data_in_k = "0001") then
                        start_add 						<= (others => '0');
                        fifo_data_out(35 downto 32) 	<= "0011";
                        fifo_data_out(31 downto 0)		<= (others => '0');
                        fifo_we							<= '1';
                        state 							<= waiting;
                    else
                        mem_data_o 			<= link_data_in(31 downto 0);
                        mem_wren_o 			<= '1';
                        mem_addr_write_o 	<= std_logic_vector(start_add);
                        start_add 			<= start_add + 1;
                    end if;
                --
            when reading =>
                    fifo_data_out 			<= "0000" & mem_data_i;
                    fifo_we					<= '1';
                    mem_addr_read_o 		<= std_logic_vector(start_add + 1);
                    start_add 				<= start_add + 1;
                    if (start_add = end_add) then
                        state 					<= end_reading;
                    end if;
                --
            when end_reading =>
                    start_add 					<= (others => '0');
                    fifo_data_out(35 downto 32) <= "0011";
                    fifo_data_out(31 downto 0)	<= (others => '0');
                    fifo_we						<= '1';
                    state 						<= waiting;
                --
            when others =>
                state <= waiting;
                --
            end case;

            --
        end if;

        --
    end if;
    end process;

end architecture;
