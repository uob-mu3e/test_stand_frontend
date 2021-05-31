odbedit -c clean
odbedit -c 'mkdir Custom'

odbedit -d Custom -c 'create STRING OnlineAna'
odbedit -d Custom -c 'create STRING Mu3eDisplay'

odbedit -c 'set Custom/OnlineAna ../mhttpd/OnlineAna.html'
odbedit -c 'set Custom/Mu3eDisplay ../mhttpd/Mu3eDisplay.html'

cd ../online
ln -sf ../packages/rootana/mhttpd .
ln -sf ../mhttpd/online_ana_setup.json .
ln -sf ../mhttpd/OnlineAna.js .
ln -sf ../mhttpd/OnlineAna.css .
ln -sf ../mhttpd/plotting_functions_jsroot.js .
ln -sf ../mhttpd/plotting_functions_nonroot.js .
ln -sf ../mhttpd/rootana_functions.js .