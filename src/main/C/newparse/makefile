ZLIB=\\\\\\\\code/zlib

MYDEFS=/D WITHGZS

RELOPTS=/O2 /I $(ZLIB) $(MYDEFS) /EHsc /D NDEBUG /Gd /GF /MT
DBGOPTS=/EHsc /I $(ZLIB) $(MYDEFS) /D _DEBUG /Gd /GF /MTd /W4 /wd4996
RELLINK=$(ZLIB)/zlib.lib /MACHINE:AMD64
DBGLINK=$(ZLIB)/zlib.lib /NODEFAULTLIB:LIBCMT /MACHINE:AMD64
CC_OPTS=$(DBGOPTS)
LINK_OPTS=$(DBGLINK)
CC=cl
CPP=cl
LD=link

include gmake.options
.SUFFIXES: 
.SUFFIXES: .c .cpp .obj .exe .lxc .flex

EXES=xmltweet.exe xmlwiki.exe trec.exe tparse.exe
OBJS=newparse.obj utils.obj gzstream.obj

.SECONDARY: xmltweet.lxc xmlwiki.lxc trec.lxc

all: $(EXES)

.flex.lxc: 
	flex -o $@ $< 

tparse.exe: gzstream.obj utils.obj tparse.obj utils.h
	$(LD) tparse.obj utils.obj gzstream.obj $(LINK_OPTS) /OUT:tparse.exe

.obj.exe: $(OBJS)
	$(LD) $< $(OBJS) $(LINK_OPTS) /OUT:"$@"

.cpp.obj: utils.h gzstream.h
	$(CPP) $(CC_OPTS) /Fo"$@" /c $<

.lxc.obj: 
	$(CPP) $(CC_OPTS) /DYY_NO_UNISTD_H /Fo"$@" /c /Tc"$<"

$(EXES): $(OBJS)

gzstream.obj: gzstream.h

debug: clean
	echo -e CC_OPTS=$(DBGOPTS)\\nLINK_OPTS=$(DBGLINK) > gmake.options

release: clean
	echo -e CC_OPTS=$(RELOPTS)\\nLINK_OPTS=$(RELLINK) > gmake.options

gmake.options:
	touch gmake.options

clean:
	rm -f $(EXES) *.obj *.lxc
