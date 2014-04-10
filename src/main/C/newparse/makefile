
include makefile.incl

ifeq ($(CC),gcc)
	include makefile.gcc
else
	include makefile.w32
endif

install: all
	mkdir -p ../../../../bin
	cp *.exe ../../../../bin