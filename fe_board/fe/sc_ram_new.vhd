--
-- SC rewrite
-- Oktober 2021, M.Mueller
--
-- - ram port has priority
-- - map upper 256 words to reg port
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sc_ram is
generic (
    RAM_ADDR_WIDTH_g : positive := 16;
    READ_DELAY_g     : positive := 6--;
);
port (
    -- RAM port (slave of sx_rx)
    i_ram_addr          : in    std_logic_vector(15 downto 0) := (others => '0');
    i_ram_re            : in    std_logic := '0';
    o_ram_rvalid        : out   std_logic;
    o_ram_rdata         : out   std_logic_vector(31 downto 0);
    i_ram_we            : in    std_logic := '0';
    i_ram_wdata         : in    std_logic_vector(31 downto 0) := (others => '0');

    -- AVS port (avalon slave of nios)
    -- address units - words
    i_avs_address       : in    std_logic_vector(15 downto 0) := (others => '0');
    i_avs_read          : in    std_logic := '0';
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic := '0';
    i_avs_writedata     : in    std_logic_vector(31 downto 0) := (others => '0');
    o_avs_waitrequest   : out   std_logic;

    -- REG port (master)
    o_reg_addr          : out   std_logic_vector(7 downto 0);
    o_reg_re            : out   std_logic;
    i_reg_rdata         : in    std_logic_vector(31 downto 0) := (others => '0');
    o_reg_we            : out   std_logic;
    o_reg_wdata         : out   std_logic_vector(31 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of sc_ram is

    signal iram_addr : std_logic_vector(15 downto 0);
    signal iram_we : std_logic;
    signal iram_rdata, iram_wdata : std_logic_vector(31 downto 0);

    type sc_req_state_type is (idle, stuff2);
    type sc_rec_state_type is (idle, stuff2);

    signal sc_req_state : sc_req_state_type := idle;
    signal sc_rec_state : sc_rec_state_type := idle;

    signal read_delay_shift_reg         : std_logic_vector(READ_DELAY_g downto 0) := (others => '0');
    signal read_delay_shift_reg_type    : std_logic_vector(READ_DELAY_g downto 0) := (others => '0'); -- 0: Arria, 1: Nios

    signal internal_ram_return_queue    : reg32_array(READ_DELAY_g downto 0);

begin

    -- request process
    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            iram_we         <= '0';
            sc_req_state    <= idle;
            sc_rec_state    <= idle;
            read_delay_shift_reg        <= (others => '0');
            read_delay_shift_reg_type   <= (others => '0');

        elsif rising_edge(i_clk) then
            -- defaults
            iram_we      <= '0';
            o_reg_we     <= '0';
            
            read_delay_shift_reg        <= read_delay_shift_reg(READ_DELAY_g downto 1) & '0';
            read_delay_shift_reg_type   <= read_delay_shift_reg_type(READ_DELAY_g downto 1) & '0';

            if(i_ram_we='1') then -- write from Arria10

                -- sc regs if addr >= 0xFF00
                if(i_ram_addr(15 downto 8)= 0xFF) then
                    o_reg_we        <= '1';
                end if;
                o_reg_wdata     <= i_ram_wdata;
                o_reg_addr      <= i_ram_addr(7 downto 0);

                -- internal mem
                iram_we         <= '1';
                iram_wdata      <= i_ram_wdata;
                iram_addr       <= i_ram_addr;

            elsif(i_ram_re='1') then -- read from Arria10
                read_delay_shift_reg(0) <= '1';

                if(i_ram_addr(15 downto 8)= 0xFF) then
                    o_reg_re        <= '1';
                end if;
                o_reg_addr      <= i_ram_addr(7 downto 0);

                -- internal mem
                iram_addr       <= i_ram_addr;

            elsif(i_avs_write='1') then -- write from nios

                -- sc regs if addr >= 0xFF00
                if(i_avs_address(15 downto 8)= 0xFF) then
                    o_reg_we        <= '1';
                end if;
                o_reg_wdata     <= i_avs_writedata;
                o_reg_addr      <= i_avs_address(7 downto 0);
                
                -- internal mem
                iram_we         <= '1';
                iram_wdata      <= i_avs_writedata;
                iram_addr       <= i_avs_address;

            elsif(i_avs_read='1') then -- read from nios
                read_delay_shift_reg(0)         <= '1';
                read_delay_shift_reg_type(0)    <= '1';

                if(i_ram_addr(15 downto 8)= 0xFF) then
                    o_reg_re            <= '1';
                end if;
                o_reg_addr      <= i_avs_address(7 downto 0);

                -- internal mem
                iram_addr       <= i_avs_address;

            end if;
        end if;
    end process;


    -- response process
    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then

        elsif rising_edge(i_clk) then
            -- defaults
            o_ram_rvalid    <= '0';
        
            -- delay internal ram by the same amount of cycles as the sc regs
            -- (avoids collisions between read response from internal ram and sc reg)
            internal_ram_return_queue => internal_ram_return_queue(READ_DELAY_g-1 downto 0) & iram_rdata;

            if(read_delay_shift_reg(READ_DELAY_g) = '1') then
                if(read_delay_shift_reg_type(READ_DELAY_g) = '0') then --respond to Arria10
                    o_ram_rvalid    <= '1';
                    o_ram_rdata     <= i_reg_rdata;
                else -- respond to nios
                    o_avs_readdata  <= i_reg_rdata;
                    -- TODO: search for correct avm_waitrequest timing
                end if;
            end if;
        end if;
    end process;


    -- internal RAM
    e_iram : entity work.ram_1r1w
    generic map (
        g_DATA_WIDTH => 32,
        g_ADDR_WIDTH => RAM_ADDR_WIDTH_g--,
    )
    port map (
        i_raddr => iram_addr(RAM_ADDR_WIDTH_g-1 downto 0),
        o_rdata => iram_rdata,
        i_rclk => i_clk,

        i_waddr => iram_addr(RAM_ADDR_WIDTH_g-1 downto 0),
        i_wdata => iram_wdata,
        i_we => iram_we,
        i_wclk => i_clk--,
    );

end architecture;
