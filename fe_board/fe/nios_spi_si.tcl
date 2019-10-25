#

add_instance spi_si altera_avalon_spi
nios_base.connect spi_si clk reset spi_control_port 0x700F0260

add_connection cpu.irq spi_si.irq
set_connection_parameter_value cpu.irq/spi_si.irq irqNumber 12

add_interface spi_si conduit end
set_interface_property spi_si EXPORTOF spi_si.external
