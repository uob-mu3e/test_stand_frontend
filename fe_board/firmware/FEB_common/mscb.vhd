library ieee;
use ieee.std_logic_1164.all;

entity mscb is
generic (
    BAUD_RATE_g : positive := 115200;
    CLK_MHZ_g : real--;
);
port (
    -- avalon slave interface
    -- - read latency 1
    i_avs_address       : in    std_logic_vector(3 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    i_rx_data           : in    std_logic;
    o_tx_data           : out   std_logic;
    o_tx_data_oe        : out   std_logic;

    o_irq               : out   std_logic;
    i_mscb_address              : in    std_logic_vector(15 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture rtl of mscb is

    signal addressing_rdata : std_logic_vector(8 downto 0);
    signal addressing_rack, addressing_rempty : std_logic;

    signal in_fifo_full :               std_logic;

    signal rx_rdata : std_logic_vector(8 downto 0);
    signal rx_rack, rx_rempty : std_logic;

    signal tx_wdata : std_logic_vector(8 downto 0);
    signal tx_we, tx_wfull : std_logic;

    signal addressing_data_out :        std_logic_vector(8 downto 0);
    signal addressing_wrreq :           std_logic;

begin

    -- memory interface
    process(i_clk)
    begin
    if rising_edge(i_clk) then
        o_avs_readdata <= X"CCCCCCCC";
        o_avs_waitrequest <= '0';

        if ( i_avs_address = X"0" and i_avs_read = '1' ) then
            o_avs_readdata <= (others => '0');
            o_avs_readdata(11 downto 0) <= '1' & in_fifo_full & addressing_rempty & addressing_rdata;
        end if;

        addressing_rack <= '0';
        tx_we <= '0';
        if ( i_avs_address = X"0" and i_avs_write = '1' ) then
            addressing_rack <= i_avs_writedata(10);
            tx_we <= i_avs_writedata(9);
            tx_wdata <= i_avs_writedata(8 downto 0);
        end if;
        --
    end if;
    end process;

    o_irq <= not addressing_rempty;

    e_uart_rx : entity work.uart_rx
    generic map (
        DATA_BITS_g => 9,
        BAUD_RATE_g => BAUD_RATE_g,
        CLK_MHZ_g => CLK_MHZ_g--,
    )
    port map (
        i_data          => i_rx_data,

        o_rdata         => rx_rdata,
        i_rack          => rx_rack,
        o_rempty        => rx_rempty,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    -- fifo addressing to nios (addressing is done, only commands intended for this mscb node)
    e_nios_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 9,
        SHOWAHEAD   => "OFF"--,
    )
    port map (
        q               => addressing_rdata,
        rdreq           => addressing_rack,
        empty           => addressing_rempty,

        data            => addressing_data_out,
        wrreq           => addressing_wrreq,
        full            => in_fifo_full,

        sclr            => not i_reset_n,
        clock           => i_clk
    );

    e_uart_tx : entity work.uart_tx
    generic map (
        DATA_BITS_g => 9,
        BAUD_RATE_g => BAUD_RATE_g,
        CLK_MHZ_g => CLK_MHZ_g--,
    )
    port map (
        o_data          => o_tx_data,
        o_data_oe       => o_tx_data_oe,

        i_wdata         => tx_wdata,
        i_we            => tx_we,
        o_wfull         => tx_wfull,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    e_mscb_addressing : entity work.mscb_addressing
    port map (
        i_data          => rx_rdata,
        o_rdreq         => rx_rack,
        i_empty         => rx_rempty,

        o_data          => addressing_data_out,
        o_wrreq         => addressing_wrreq,

        i_address       => i_mscb_address,

        i_reset         => not i_reset_n,
        i_clk           => i_clk--,
    );

end architecture;
