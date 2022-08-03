odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/ -c "mkdir VPDAC_scan"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan -c "mkdir Input"

odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Input -c "create INT32 HV_voltage"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Input -c "create INT32 start_value"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Input -c "create INT32 step"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Input -c "create INT32 stop_value"

odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/ -c "mkdir Output"

odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/ -c "create INT32 ladder_temp"

for i in {0..119}
do 
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Output/ -c "mkdir $i"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Output/$i -c "create INT32 values[32]"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/VPDAC_scan/Output/$i -c "create FLOAT current[32]"    
done
