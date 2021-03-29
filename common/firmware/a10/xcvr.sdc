#

set_min_delay -to {xcvr_a10:*|av_ctrl.readdata[*]} -100
set_max_delay -to {xcvr_a10:*|av_ctrl.readdata[*]} 100
set_min_delay -to {*|xcvr_a10:*|av_ctrl.readdata[*]} -100
set_max_delay -to {*|xcvr_a10:*|av_ctrl.readdata[*]} 100

set_min_delay -to {xcvr_enh:*|av_ctrl.readdata[*]} -100
set_max_delay -to {xcvr_enh:*|av_ctrl.readdata[*]} 100
set_min_delay -to {*|xcvr_enh:*|av_ctrl.readdata[*]} -100
set_max_delay -to {*|xcvr_enh:*|av_ctrl.readdata[*]} 100
