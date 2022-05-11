odbedit -d /Sequencer/State -c "set Path ${PWD}"

odbedit -d /Equipment/PixelsCentral/QCTests -c "mkdir Ladder_FEB_level"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level -c "mkdir Link_Quali_test"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test -c "mkdir Input"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test -c "mkdir Output"

odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test -c "create INT32 feb_id"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test -c "create INT32 hameg_id"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test -c "create INT32 hameg_channel_id_1"
odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test -c "create INT32 hameg_channel_id_2"

odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Input -c "create INT32 waiting_time"

odbedit -d /Sequencer/State -c "set Path ${PWD}"

for i in {0..11}
do
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i -c "mkdir Scan"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i -c "create INT32 opt_VPVCO"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i -c "create INT32 opt_VNVCO"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i -c "create INT32 ErrorRate"

    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i/Scan -c "create INT32 VPVCO[30]"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i/Scan -c "create INT32 error_rate[30]"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i/Scan -c "create INT32 error_rate_linkA[30]"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i/Scan -c "create INT32 error_rate_linkB[30]"
    odbedit -d /Equipment/PixelsCentral/QCTests/Ladder_FEB_level/Link_Quali_test/Output/$i/Scan -c "create INT32 error_rate_linkC[30]"

done