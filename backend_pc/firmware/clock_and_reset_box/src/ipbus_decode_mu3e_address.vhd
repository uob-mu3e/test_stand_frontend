library IEEE;
use IEEE.STD_LOGIC_1164.all;
use ieee.numeric_std.all;

package ipbus_decode_mu3e_address is

  constant IPBUS_SEL_WIDTH: positive := 5; -- Should be enough for now?
  subtype ipbus_sel_t is std_logic_vector(IPBUS_SEL_WIDTH - 1 downto 0);
  function ipbus_sel_mu3e_address(addr : in std_logic_vector(31 downto 0)) return ipbus_sel_t;

-- START automatically  generated VHDL the Fri Jan 18 11:03:25 2019
  constant N_SLV_FIFO_REG_OUT: integer := 0;
  constant N_SLV_FIFO_REG_OUT_CHARISK: integer := 1;
  constant N_SLV_FIFO_REG_IN: integer := 2;
  constant N_SLV_CTRL_REG: integer := 3;
  constant N_SLV_I2C: integer := 4;
  constant N_SLAVES: integer := 5;
-- END automatically generated VHDL


end ipbus_decode_mu3e_address;

package body ipbus_decode_mu3e_address is

  function ipbus_sel_mu3e_address(addr : in std_logic_vector(31 downto 0)) return ipbus_sel_t is
    variable sel: ipbus_sel_t;
  begin

-- START automatically  generated VHDL the Fri Jan 18 11:03:25 2019
    if    std_match(addr, "----------------------------000-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_FIFO_REG_OUT, IPBUS_SEL_WIDTH)); -- fifo_reg_out / base 0x00000000 / mask 0x0000000e
    elsif std_match(addr, "----------------------------001-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_FIFO_REG_OUT_CHARISK, IPBUS_SEL_WIDTH)); -- fifo_reg_out_charisk / base 0x00000002 / mask 0x0000000e
    elsif std_match(addr, "----------------------------010-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_FIFO_REG_IN, IPBUS_SEL_WIDTH)); -- fifo_reg_in / base 0x00000004 / mask 0x0000000e
    elsif std_match(addr, "----------------------------011-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_CTRL_REG, IPBUS_SEL_WIDTH)); -- ctrl_reg / base 0x00000006 / mask 0x0000000e
    elsif std_match(addr, "----------------------------1---") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_I2C, IPBUS_SEL_WIDTH)); -- i2c / base 0x00000008 / mask 0x00000008
-- END automatically generated VHDL

    else
        sel := ipbus_sel_t(to_unsigned(N_SLAVES, IPBUS_SEL_WIDTH));
    end if;

    return sel;

  end function ipbus_sel_mu3e_address;

end ipbus_decode_mu3e_address;
