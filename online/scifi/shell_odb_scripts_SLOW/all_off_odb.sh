#!/bin/bash
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set ext_trig_mode                   n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set ext_trig_endtime_sign           n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set ext_trig_offset                 0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set ext_trig_endtime                0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set gen_idle                        y"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set ms_debug                        n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set prbs_debug                      n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set prbs_single                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set sync_ch_rst                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set disable_coarse                  n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set pll_setcoarse                   n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set short_event_mode                n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set pll_envomonitor                 n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set pll_lol_dbg                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Global" -c "set en_ch_evt_cnt                   n"



for i in {0..15}
do
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnpfc                           63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnpfc_offset                    3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnpfc_scale                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vncnt                           63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vncnt_offset                    3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vncnt_scale                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnvcobuffer                     63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnvcobuffer_offset              3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnvcobuffer_scale               n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnd2c                           63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnd2c_offset                    3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnd2c_scale                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnpcp                           63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnpcp_offset                    3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnpcp_scale                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnhitlogic                      63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnhitlogic_offset               3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnhitlogic_scale                n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vncntbuffer                     63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vncntbuffer_offset              3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vncntbuffer_scale               n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnvcodelay                      63"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnvcodelay_offset               3"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set vnvcodelay_scale                n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set latchbias                       0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set ms_limits                       0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set ms_switch_sel                   n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set amon_en                         n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set amon_dac                        0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set dmon_1_en                       n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set dmon_1_dac                      0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set dmon_2_en                       n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set dmon_2_dac                      0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set lvds_tx_vcm                     155"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set lvds_tx_bias                    0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_lower_rx_ena          n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_lower_tx_ena          n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_lower_tx_vdac         0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_lower_tx_idac         0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_mat_xbl                    0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_mat_xbu                    0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_upper_rx_ena          n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_upper_tx_ena          n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_upper_tx_vdac         0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/TDCs/${i}" -c "set coin_xbar_upper_tx_idac         0"
done

for i in {0..511}
do
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set mask                            n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set recv_all                        n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set tthresh                         0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set tthresh_sc                      0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set ethresh                         255"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set ebias                           0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set sipm                            0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set sipm_sc                         0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set inputbias                       0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set inputbias_sc                    0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set pole                            0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set pole_sc                         0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set ampcom                          0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set ampcom_sc                       0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set cml                             0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set cml_sc                          0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set amonctrl                        0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set comp_spi                        0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set coin_mat                        0"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set tdctest_n                       n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set sswitch                         n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set delay                           n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set pole_en_n                       n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set energy_c_en                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set energy_r_en                     n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set cm_sensing_high_r               n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set amon_en_n                       n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set edge                            n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set edge_cml                        n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set dmon_en                         n"
odbedit -d "/Equipment/SciFi/Settings/ASICs/Channels/${i}" -c "set dmon_sw                         n"
done
