include Makefile.incl

MKL_OBJS=

CUDA_OBJS=BIDMach_CUMACH.$(OBJ) Samplers.$(OBJ) Dtree.$(OBJ) Devel.$(OBJ) GLM.$(OBJ) HashMult.$(OBJ) \
          Logger.$(OBJ) JNIUtils.$(OBJ) PointerUtils.$(OBJ)

.SUFFIXES: .$(OBJ) .c .cpp .cu

all: $(LIBPREPEND)bidmachcuda$(LIBAPPEND) 
#all: $(LIBPREPEND)bidmachmkl$(LIBAPPEND) 

$(LIBPREPEND)bidmachmkl$(LIBAPPEND): $(MKL_OBJS)
	$(LD) $(LDFLAGS) $(MKL_OBJS) $(MKL_LIBS) $(OUTFLG)$@

$(LIBPREPEND)bidmachcuda$(LIBAPPEND): $(CUDA_OBJS)
	$(GLD) $(LDFLAGS) $(CUDA_OBJS) $(CUDA_LIBS) $(OUTFLG)$@

%.$(OBJ) : %.c
	$(CC) $(CFLAGS) $(LAPACK_INCLUDES) $*.c

%.$(OBJ) : %.cpp
	$(GCC) $(CPPFLAGS) $(LAPACK_INCLUDES) $*.cpp

#Dtree.$(OBJ) : Dtree.cu
#	$(NVCC) $(NVCCFLAGS) -Xptxas -v Dtree.cu

%.$(OBJ) : %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

install: all
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) $(INSTALL_DIR)
#	cp $(LIBPREPEND)bidmachmkl$(LIBAPPEND)  $(INSTALL_DIR)

clean:
	rm -f *.$(OBJ) *$(LIBAPPEND) *.pdb *.exp *.lib

distclean: clean
	rm -f  *.jnilib Makefile.incl

