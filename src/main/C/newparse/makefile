include makefile.incl

RELOPTS=$(CFLAGS) -O2 -DNDEBUG -Wno-deprecated -I.
DBGOPTS=$(CFLAGS) -O2 -DDEBUG -g -I.
RELLINK=-lz
DBGLINK==-lz
CC_OPTS=$(DBGOPTS)
LINK_OPTS=$(DBGLINK)
CC=gcc
CPP=g++
LD=g++
include gmake.options
.SUFFIXES: 
.SUFFIXES: .c .cpp .o .exe .lxc .flex

EXES=xmltweet.exe xmlwiki.exe trec.exe tparse.exe
OBJS=newparse.o utils.o gzstream.o

.SECONDARY: xmltweet.lxc xmlwiki.lxc trec.lxc

all: $(EXES)

.flex.lxc: 
	flex -o $@ $< 

tparse.exe: gzstream.o utils.o tparse.o utils.h
	$(LD) -o tparse.exe tparse.o utils.o gzstream.o $(LINK_OPTS)

.o.exe: $(OBJS)
	$(LD) -o $@ $(OBJS) $< $(LINK_OPTS)

.cpp.o: utils.h gzstream.h
	$(CPP) $(CC_OPTS) -o $@ -c $<

.lxc.o: 
	$(CPP) $(CC_OPTS) -DYY_NO_UNISTD_H -o $@ -c -x c++ $<

$(EXES): $(OBJS)

gzstream.o: gzstream.h

debug: clean
	echo -e CC_OPTS=$(DBGOPTS)\\nLINK_OPTS=$(DBGLINK) > gmake.options

release: clean
	echo -e CC_OPTS=$(RELOPTS)\\nLINK_OPTS=$(RELLINK) > gmake.options

gmake.options:
	touch gmake.options

clean:
	rm -f $(EXES) *.o *.lxc
