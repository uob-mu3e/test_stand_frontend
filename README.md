# mu3e general repository for online development

 - firmware for all subsystems
 - midas frontend for subsystems
 - software
<<<<<<< HEAD
=======

[TODO] Link to wiki page

Top level functions as midas "online":
    - .SHM files etc.
    - exptab


Devices:
one device/location = one directory
    - Common
        - firmware (collection of vhd files)
        - everything mudaq + dma + slow control
        - mudaq driver 

    - Backend (PC)
        - TBD 
        - software
          - clock board
          - xcontrol

    - Clock (board)
        - Firmware
    

    - Switching (pc)
        - a10_board (lhcb board)
          - firmware
        - Midas Frontend
        - Software:
            - libmutrig
            - future mupix control software

    - Farm (pc)
        - Midas Frontend
        currently two frontends with equipments "stream" and "Stream"
        - a10_board
           - firmware

    - FEB (board)
        - firmware
            - software (i.e. nios)


Modules:
external packages
    - MIDAS
    - MXML
    - MSCB

mhttpd:
    - config for web server
    - custom pages

>>>>>>> origin/master
