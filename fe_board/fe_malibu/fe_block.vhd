library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fe_block is
generic (
    FPGA_ID_g : std_logic_vector(15 downto 0) := X"0000";
    SPI_SLAVES : positive := 1--;
);
port (
    i_i2c_scl   : in    std_logic;
    o_i2c_scl   : out   std_logic;
    i_i2c_sda   : in    std_logic;
    o_i2c_sda   : out   std_logic;

    i_spi_miso  : in    std_logic;
    o_spi_mosi  : out   std_logic;
    o_spi_sclk  : out   std_logic;
    o_spi_ss_n  : out   std_logic_vector(15 downto 0);



    -- MSCB interface
    i_mscb_data : in    std_logic;
    o_mscb_data : out   std_logic;
    o_mscb_oe   : out   std_logic;



    -- QSFP links
    i_qsfp_rx       : in    std_logic_vector(3 downto 0);
    o_qsfp_tx       : out   std_logic_vector(3 downto 0);
    i_qsfp_clk      : in    std_logic;

    i_fifo_data     : in    std_logic_vector(31 downto 0);
    i_fifo_empty    : in    std_logic;
    o_fifo_rack     : out   std_logic;



    -- POD links (reset system)
    i_pod_rx        : in    std_logic_vector(3 downto 0);
    o_pod_tx        : out   std_logic_vector(3 downto 0);
    i_pod_clk       : in    std_logic;



    -- avalon master
    -- address space - 64 bytes (16 words)
    -- address units - words
    -- read latency - 1
    o_avm_address       : out   std_logic_vector(13 downto 0);
    o_avm_read          : out   std_logic;
    i_avm_readdata      : in    std_logic_vector(31 downto 0);
    o_avm_write         : out   std_logic;
    o_avm_writedata     : out   std_logic_vector(31 downto 0);
    i_avm_waitrequest   : in    std_logic;

    -- SC RAM
    i_sc_ram_address    : in    std_logic_vector(15 downto 0);
    o_sc_ram_rdata      : out   std_logic_vector(31 downto 0);
    i_sc_ram_wdata      : in    std_logic_vector(31 downto 0);
    i_sc_ram_we         : in    std_logic;

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of fe_block is

begin

end architecture;
