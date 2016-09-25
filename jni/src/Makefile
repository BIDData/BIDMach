include Makefile.incl

CUDA_OBJS=BIDMach_CUMACH.$(OBJ) Samplers.$(OBJ) Dtree.$(OBJ) Devel.$(OBJ) GLM.$(OBJ) HashMult.$(OBJ) \
          DNN.$(OBJ) DNN127.$(OBJ) Logger.$(OBJ) JNIUtils.$(OBJ) PointerUtils.$(OBJ)

CPU_OBJS=BIDMach_CPUMACH.$(OBJ)

.SUFFIXES: .$(OBJ) .c .cpp .cu

all: $(LIBPREPEND)bidmachcuda$(LIBAPPEND)  $(LIBPREPEND)bidmachcpu$(LIBAPPEND) 

cudalib: $(LIBPREPEND)bidmachcuda$(LIBAPPEND)

cpulib: $(LIBPREPEND)bidmachcpu$(LIBAPPEND) 

$(LIBPREPEND)bidmachcpu$(LIBAPPEND): $(CPU_OBJS)
	$(LD) $(LDFLAGS) $(CPU_OBJS) $(CPU_LIBS) $(OUTFLG)$@

$(LIBPREPEND)bidmachcuda$(LIBAPPEND): $(CUDA_OBJS)
	$(GLD) $(LDFLAGS) $(CUDA_OBJS) $(CUDA_LIBS) $(OUTFLG)$@

%.$(OBJ) : %.c
	$(CC) $(CFLAGS) $(LAPACK_INCLUDES) $*.c

%.$(OBJ) : %.cpp
	$(GCC) $(CPPFLAGS) $(LAPACK_INCLUDES) $*.cpp

DNN63.$(OBJ) : DNN63.cu
	$(NVCC) --maxrregcount=63 $(NVCCFLAGS) -Xptxas -v DNN63.cu

DNN127.$(OBJ) : DNN127.cu
	$(NVCC) --maxrregcount=127 $(NVCCFLAGS) DNN127.cu
#	$(NVCC) --maxrregcount=127 $(NVCCFLAGS) -Xptxas -v DNN127.cu

%.$(OBJ) : %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

install: installcudalib installcpulib

installcudalib: $(INSTALLDIR)/$(LIBPREPEND)bidmachcuda$(LIBAPPEND)

installcpulib: $(INSTALLDIR)/$(LIBPREPEND)bidmachcpu$(LIBAPPEND)

$(INSTALLDIR)/$(LIBPREPEND)bidmachcuda$(LIBAPPEND): $(LIBPREPEND)bidmachcuda$(LIBAPPEND)
	mkdir -p $(INSTALL_DIR)
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) $(INSTALL_DIR)

$(INSTALLDIR)/$(LIBPREPEND)bidmachcpu$(LIBAPPEND): $(LIBPREPEND)bidmachcpu$(LIBAPPEND)
	mkdir -p $(INSTALL_DIR)
	cp $(LIBPREPEND)bidmachcpu$(LIBAPPEND) $(INSTALL_DIR)
	cp $(LIBDIR)/libiomp5$(LIBIOMPAPPEND)  $(INSTALL_DIR)

jars: cudajar cpujar

cudajar: $(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-$(OS)-$(MARCH).jar

cpujar: $(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-cpu-$(OS)-$(MARCH).jar

$(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-$(OS)-$(MARCH).jar: $(LIBPREPEND)bidmachcuda$(LIBAPPEND)
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) $(LIBDIR)
	jar cf $(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-$(OS)-$(MARCH).jar \
	-C $(BIDMACH_ROOT) lib/$(LIBPREPEND)bidmachcuda$(LIBAPPEND)
	rm $(LIBDIR)/$(LIBPREPEND)bidmachcuda$(LIBAPPEND)

$(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-cpu-$(OS)-$(MARCH).jar: $(LIBPREPEND)bidmachcpu$(LIBAPPEND)
	cp $(LIBPREPEND)bidmachcpu$(LIBAPPEND) $(LIBDIR)
	jar cf $(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-cpu-$(OS)-$(MARCH).jar \
	-C $(BIDMACH_ROOT) lib/$(LIBPREPEND)bidmachcpu$(LIBAPPEND)
	jar uf $(BIDMACH_ROOT)/target/BIDMach-$(VERSION)-cpu-$(OS)-$(MARCH).jar \
	-C $(BIDMACH_ROOT) lib/libiomp5$(LIBIOMPAPPEND)
	rm $(LIBDIR)/$(LIBPREPEND)bidmachcpu$(LIBAPPEND)

upload: $(LIBPREPEND)bidmachcpu$(LIBAPPEND) $(LIBPREPEND)bidmachcuda$(LIBAPPEND)
	if [ -d /biddata/lib ]; then \
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) /biddata/lib/$(DEVVERSION) ; \
	cp $(LIBPREPEND)bidmachcpu$(LIBAPPEND)  /biddata/lib/$(DEVVERSION) ; else \
	scp $(LIBPREPEND)bidmach*$(LIBAPPEND) login.eecs.berkeley.edu:biddata/lib/$(DEVVERSION) ; fi

clean:
	rm -f *.$(OBJ) *$(LIBAPPEND) *.pdb *.exp *.lib

distclean: clean
	rm -f  *.jnilib Makefile.incl

cleanres:
	rm $(BIDMACH_ROOT)/src/main/resources/lib/*
