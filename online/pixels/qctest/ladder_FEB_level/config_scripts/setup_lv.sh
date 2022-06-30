odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/ -c "mkdir LV_power_on_test"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_power_on_test -c "mkdir Output"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_power_on_test -c "mkdir Input"

odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_power_on_test/Input -c "create INT32 order"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_power_on_test/Input -c "create INT32 start_condition"


odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/ -c "mkdir LV_map"

for i in {0..38}
do
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_power_on_test/Output -c "mkdir $i"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_power_on_test/Output/$i -c "create FLOAT current_increase[3]"

done

for i in {0..37}
do
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_map/ -c "mkdir $i"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_map/$i -c "create INT32 hameg_id"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/LV_map/$i -c "create INT32 channel_id"
done
