
include makefile.incl

ifeq ($(CC),gcc)
	include makefile.gcc
else
	include makefile.w32
endif

install: all
	mkdir -p ../../../../cbin
	cp *.exe ../../../../cbin
	mkdir -p ../../../../src/main/resources/cbin
	cp *.exe ../../../../src/main/resources/cbin