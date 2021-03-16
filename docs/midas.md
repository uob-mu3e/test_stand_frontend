# MIDAS

## Compiling and starting

```console
$ cd modules
$ git submodule update --init --recursive
$ cd ../../
$ mdkir build
$ cd build
$ cmake ..
$ make
$ make install
$ source set_env.sh
$ start_daq.sh
```
If you are not using the full clock system, you should replace the last line with

```console
$ start_daq.sh CRFE_BYPASS
```

If you do it the first time you also need to generate a SSL certificate and a password. This you need to do after source set_env.sh.
```console
$ openssl req -new -nodes -newkey rsa:2048 -sha256 -out ssl_cert.csr -keyout ssl_cert.key
$ openssl x509 -req -days 365 -sha256 -in ssl_cert.csr -signkey ssl_cert.key -out ssl_cert.pem
$ cat ssl_cert.key >> ssl_cert.pem
$ touch htpasswd.txt
$ htdigest htpasswd.txt Default midas
```

If you want to use the analyzer you also need to do the following:
```console
$ cd build 
$ ./modules/analyzer/frontends/dummy_fe (optional - if you want a dummy)
$ ./modules/analyzer/analyzer_mu3e -EDefault -Hlocalhost -R8088
```

If you want to use the Event Display you also need to do the following:
```console
$ cd modules/analyzer/packages/mu3edisplay 
$ npm install
$ ./node_modules/.bin/webpack
$ cd modules/analyzer/analyzer
$ python event_api.py (API at localhost:5000)
```

Now you need to load / make the switching board driver. You also need to remove and rescan your pci devices if you uploaded the firmware to the arria10 after starting up the pc. Be careful here that you make sure no program is using the arria10. This can lead to problems if you want to load the driver again. You also need to check which device number is the arria10. You can do this by calling sudo lspci.
```console
$ cd ../common/kerneldriver
$ make
$ su
$ echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove
$ echo 1 > /sys/bus/pci/rescan
$ ./load_mudaq.sh
```
Now you can start the switch_fe and the Switching equipment should show up on your midas status page.


![Switching Page] <https://bitbucket.org/mu3e/online/src/lab_setup_arria10/lab_setup.pdf>
