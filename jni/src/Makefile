include Makefile.incl

CUDA_OBJS=BIDMach_CUMACH.$(OBJ) Samplers.$(OBJ) Dtree.$(OBJ) Devel.$(OBJ) GLM.$(OBJ) HashMult.$(OBJ) \
          DNN.$(OBJ) DNN127.$(OBJ) Logger.$(OBJ) JNIUtils.$(OBJ) PointerUtils.$(OBJ)

CPU_OBJS=BIDMach_CPUMACH.$(OBJ)

.SUFFIXES: .$(OBJ) .c .cpp .cu

all: $(LIBPREPEND)bidmachcuda$(LIBAPPEND)  $(LIBPREPEND)bidmachcpu$(LIBAPPEND) 

cudalib: $(LIBPREPEND)bidmachcuda$(LIBAPPEND)

cpulib: $(LIBPREPEND)bidmachcpu$(LIBAPPEND) 

$(LIBPREPEND)bidmachcpu$(LIBAPPEND): $(CPU_OBJS)
	$(LD) $(LDFLAGS) $(CPU_OBJS) $(MKL_LIBS) $(OUTFLG)$@

$(LIBPREPEND)bidmachcuda$(LIBAPPEND): $(CUDA_OBJS)
	$(GLD) $(LDFLAGS) $(CUDA_OBJS) $(CUDA_LIBS) $(OUTFLG)$@

%.$(OBJ) : %.c
	$(CC) $(CFLAGS) $(LAPACK_INCLUDES) $*.c

%.$(OBJ) : %.cpp
	$(GCC) $(CPPFLAGS) $(LAPACK_INCLUDES) $*.cpp

DNN63.$(OBJ) : DNN63.cu
	$(NVCC) --maxrregcount=63 $(NVCCFLAGS) -Xptxas -v DNN63.cu

DNN127.$(OBJ) : DNN127.cu
	$(NVCC) --maxrregcount=127 $(NVCCFLAGS) -Xptxas -v DNN127.cu

%.$(OBJ) : %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

install: all
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) $(INSTALL_DIR)
	cp $(LIBPREPEND)bidmachcpu$(LIBAPPEND)  $(INSTALL_DIR)

installcudalib: cudalib
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) $(INSTALL_DIR)

installcpulib: cpulib
	cp $(LIBPREPEND)bidmachcpu$(LIBAPPEND)  $(INSTALL_DIR)

upload: all
	if [ -d /biddata/lib ]; then \
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) /biddata/lib ; \
	cp $(LIBPREPEND)bidmachcpu$(LIBAPPEND)  /biddata/lib ; else \
	scp $(LIBPREPEND)bidmach*$(LIBAPPEND) bid01.ist.berkeley.edu:biddata/lib ; fi

clean:
	rm -f *.$(OBJ) *$(LIBAPPEND) *.pdb *.exp *.lib

distclean: clean
	rm -f  *.jnilib Makefile.incl

