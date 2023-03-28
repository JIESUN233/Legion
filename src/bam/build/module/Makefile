
ifneq ($(KERNELRELEASE),)
	src := /home/sunjie/bam/module
	obj-m := libnvm.o
	libnvm-objs := pci.o list.o ctrl.o map.o
	ccflags-y += -D_CUDA -I/usr/src/nvidia-515.43.04/nvidia -I/home/sunjie/bam/src -I/home/sunjie/bam/src/linux
	KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-515.43.04/Module.symvers
else

.PHONY: default reload unload load clean install

default:
	$(MAKE) -C /lib/modules/5.4.0-128-generic/build M=/home/sunjie/bam/build/module modules

clean:
	$(MAKE) -C /lib/modules/5.4.0-128-generic/build M=/home/sunjie/bam/build/module clean

reload: unload load

unload:
	-rmmod libnvm.ko

load:
	insmod libnvm.ko max_num_ctrls=64

install: default
	$(MAKE) -C /lib/modules/5.4.0-128-generic/build M=/home/sunjie/bam/build/module modules_install
	#$(MAKE) INSTALL_MOD_DIR=libnvm -C /lib/modules/5.4.0-128-generic/build M=/home/sunjie/bam/build/module modules_install

endif

