source midas.sh
mlogger -D
mhttpd -D
msequencer -D
cd online/build 
./frontend
cd ..
cd ..
killall mhttpd
killall mlogger
killall msequencer
