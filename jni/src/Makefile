include Makefile.incl

MKL_OBJS=

CUDA_OBJS=BIDMach_CUMACH.$(OBJ) Samplers.$(OBJ) Dtree.$(OBJ) Devel.$(OBJ) GLM.$(OBJ) HashMult.$(OBJ) DNN63.$(OBJ) DNN127.$(OBJ) \
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

DNN63.$(OBJ) : DNN63.cu
	$(NVCC) --maxrregcount=127 $(NVCCFLAGS) -Xptxas -v DNN63.cu

DNN127.$(OBJ) : DNN127.cu
	$(NVCC) --maxrregcount=128 -c -use_fast_math  -I/jni/include -gencode arch=compute_35,code=sm_35           -gencode arch=compute_50,code=sm_50           --machine 64  -Xcompiler "/EHsc /W3 /nologo /O2 /Zi  /MT" -Xptxas -v DNN127.cu

%.$(OBJ) : %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

install: all
	cp $(LIBPREPEND)bidmachcuda$(LIBAPPEND) $(INSTALL_DIR)
#	cp $(LIBPREPEND)bidmachmkl$(LIBAPPEND)  $(INSTALL_DIR)

clean:
	rm -f *.$(OBJ) *$(LIBAPPEND) *.pdb *.exp *.lib

distclean: clean
	rm -f  *.jnilib Makefile.incl

