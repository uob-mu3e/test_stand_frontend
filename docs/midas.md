# MIDAS

## Compiling with your own MIDAS version
This repository requires the user to set up midas on their PC.
Refer to the [MIDAS documentation](https://midas.triumf.ca/MidasWiki/index.php/Quickstart_Linux) for help with that.
When cmake fails because it could not find MIDAS, you need to tell it where it is.
The standard way to do so is via the `CMAKE_PREFIX_PATH` variable including the root directory of MIDAS.
Run CMake with the option `-DCMAKE_PREFIX_PATH=/path/to/your/midas` option, or set the environment variable:

```
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/path/to/your/midas"
```

## Running with MIDAS
You need to set up your environment to run the DAQ.
Most of this is done in the script `set_env.sh`
This includes:

* Setting the environment variable `MIDASSYS` to where your MIDAS lives (same as `CMAKE_PREFIX_PATH` above)
* Creating an [experiment table](https://midas.triumf.ca/MidasWiki/index.php/Quickstart_Linux#Create_the_Experiment_file_exptab) and setting the environment variable `MIDAS_EXPTAB` to point to it.
* The content should look something like this (note that this repository is called *online* and there is a directory called *online*. So the onlinedir typically is something like `/home/mu3e/gitrepos/online/online`.

```
#Experiment   onlinedir                  user
Mu3e          /path/to/this/repo/online username
```


## Some useful notes
If you want to run MIDAS with SSL, you need to generate an SSL certificate and a password.
```console
$ openssl req -new -nodes -newkey rsa:2048 -sha256 -out ssl_cert.csr -keyout ssl_cert.key
$ openssl x509 -req -days 365 -sha256 -in ssl_cert.csr -signkey ssl_cert.key -out ssl_cert.pem
$ cat ssl_cert.key >> ssl_cert.pem
$ touch htpasswd.txt
$ htdigest htpasswd.txt Mu3e Mu3e
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
