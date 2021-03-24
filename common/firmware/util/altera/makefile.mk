#
# author : Alexandr Kozlinskiy
#

.DEFAULT_GOAL := all

.ONESHELL :

ifndef QUARTUS_ROOTDIR
    $(error QUARTUS_ROOTDIR is undefined)
endif

ifeq ($(PREFIX),)
    override PREFIX := generated
endif

ifeq ($(SOF),)
    SOF := output_files/top.sof
endif

ifeq ($(NIOS_SOPCINFO),)
    NIOS_SOPCINFO := $(PREFIX)/nios.sopcinfo
endif

BSP_SCRIPT := software/hal_bsp.tcl
SRC_DIR := software/app_src

ifeq ($(BSP_DIR),)
    BSP_DIR := $(PREFIX)/software/hal_bsp
endif

ifeq ($(APP_DIR),)
    APP_DIR := $(PREFIX)/software/app
endif

QSYS_TCL_FILES := $(filter %.tcl,$(IPs))
QSYS_FILES := $(patsubst %.tcl,$(PREFIX)/%.qsys,$(QSYS_TCL_FILES))
SOPC_FILES := $(patsubst %.qsys,%.sopcinfo,$(QSYS_FILES))
QMEGAWIZ_XML_FILES := $(filter %.vhd.qmegawiz,$(IPs))
QMEGAWIZ_VHD_FILES := $(patsubst %.vhd.qmegawiz,$(PREFIX)/%.vhd,$(QMEGAWIZ_XML_FILES))

all : $(PREFIX)/include.qip

$(PREFIX) :
	mkdir -pv $(PREFIX)
	[ -e $(PREFIX)/util ] || ln -snv --relative -T util $(PREFIX)/util

.PHONY : $(PREFIX)/components_pkg.vhd
$(PREFIX)/components_pkg.vhd : $(PREFIX) $(SOPC_FILES) $(QMEGAWIZ_VHD_FILES)
	( cd $(PREFIX) ; ./util/altera/components_pkg.sh )

$(PREFIX)/include.qip : $(PREFIX)/components_pkg.vhd $(QSYS_FILES)
	# components package
	echo "set_global_assignment -name VHDL_FILE [ file join $$::quartus(qip_path) \"components_pkg.vhd\" ]" > $@
	# add qsys *.qsys files
	for file in $(QSYS_FILES) ; do \
	    echo "set_global_assignment -name QSYS_FILE [ file join $$::quartus(qip_path) \"$$(realpath -m --relative-to=$(PREFIX) -- $$file)\" ]" >> $@ ; \
	done
	# add qmegawiz *.qip files
	for file in $(patsubst %.vhd,%.qip,$(QMEGAWIZ_VHD_FILES)) ; do \
	    echo "set_global_assignment -name QIP_FILE [ file join $$::quartus(qip_path) \"$$(realpath -m --relative-to=$(PREFIX) -- $$file)\" ]" >> $@ ; \
	done

device.tcl :
	touch $@

$(PREFIX)/%.vhd : %.vhd.qmegawiz
	./util/altera/qmegawiz.sh $< $@

$(PREFIX)/%.qsys : %.tcl device.tcl
	./util/altera/tcl2qsys.sh $< $@

$(PREFIX)/%.sopcinfo : $(PREFIX)/%.qsys
	./util/altera/qsys-generate.sh $<

.PHONY : flow
flow : all
	./util/altera/flow.sh

.PHONY : sof2flash
sof2flash :
	sof2flash --pfl --programmingmode=PS \
        --optionbit=0x00030000 \
        --input="$(SOF)" \
        --output="$(SOF).flash" --offset=0x02B40000
	objcopy -Isrec -Obinary $(SOF).flash $(SOF).bin

.PHONY : pgm
pgm : $(SOF)
	quartus_pgm -m jtag -c $(CABLE) --operation="p;$(SOF)"

.PRECIOUS : $(BSP_DIR)
$(BSP_DIR) : $(BSP_SCRIPT) $(NIOS_SOPCINFO)
	mkdir -p $(BSP_DIR)
	nios2-bsp-create-settings \
	    --type hal --script $(SOPC_KIT_NIOS2)/sdk2/bin/bsp-set-defaults.tcl \
	    --sopc $(NIOS_SOPCINFO) --cpu-name cpu \
	    --bsp-dir $(BSP_DIR) --settings $(BSP_DIR)/settings.bsp --script $(BSP_SCRIPT)

bsp : $(BSP_DIR)

.PRECIOUS : $(APP_DIR)/main.elf
.PHONY : $(APP_DIR)/main.elf
$(APP_DIR)/main.elf : $(SRC_DIR)/* $(BSP_DIR)
	nios2-app-generate-makefile \
	    --set ALT_CFLAGS "-Wextra -Wformat=0 -pedantic -std=c++14" \
	    --bsp-dir $(BSP_DIR) --app-dir $(APP_DIR) --src-dir $(SRC_DIR)
	$(MAKE) -C $(APP_DIR) clean
	$(MAKE) -C $(APP_DIR)
	nios2-elf-objcopy $(APP_DIR)/main.elf -O srec $(APP_DIR)/main.srec
	# generate flash image (srec)
	( cd $(APP_DIR) ; make mem_init_generate )

.PHONY : app
app : $(APP_DIR)/main.elf

.PHONY : app_flash
app_flash :
	nios2-flash-programmer -c $(CABLE) --base=0x0 $(APP_DIR)/main.flash

.PHONY : flash
flash : app_flash
	nios2-flash-programmer -c $(CABLE) --base=0x0 $(SOF).flash

.PHONY : app_upload
app_upload : app
	nios2-gdb-server -c $(CABLE) -r -w 1 -g $(APP_DIR)/main.srec

.PHONY : terminal
terminal :
	nios2-terminal -c $(CABLE)
