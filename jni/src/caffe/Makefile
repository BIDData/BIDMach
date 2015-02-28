include Makefile.incl

CPU_OBJS=cbin/bidmach_caffe.$(OBJ) \
  cbin/syncedmem.$(OBJ) cbin/common.$(OBJ) \
  cbin/blob.$(OBJ) cbin/net.$(OBJ) cbin/solver.$(OBJ) \
  cbin/layer_factory.$(OBJ) cbin/data_transformer.$(OBJ) \
  cbin/math_functions.$(OBJ) cbin/im2col.$(OBJ) cbin/io.$(OBJ) \
  cbin/upgrade_proto.$(OBJ) cbin/insert_splits.$(OBJ) \
  cbin/caffe.pb.$(OBJ) cbin/caffe_pretty_print.pb.$(OBJ) \
  cbin/absval_layer.$(OBJ) cbin/accuracy_layer.$(OBJ) cbin/argmax_layer.$(OBJ) \
  cbin/bnll_layer.$(OBJ) cbin/concat_layer.$(OBJ) cbin/base_data_layer.$(OBJ) \
  cbin/contrastive_loss_layer.$(OBJ) cbin/conv_layer.$(OBJ) cbin/cudnn_conv_layer.$(OBJ) \
  cbin/cudnn_pooling_layer.$(OBJ) cbin/cudnn_relu_layer.$(OBJ) cbin/cudnn_sigmoid_layer.$(OBJ) \
  cbin/cudnn_softmax_layer.$(OBJ) cbin/cudnn_tanh_layer.$(OBJ) \
  cbin/dropout_layer.$(OBJ) cbin/dummy_data_layer.$(OBJ) cbin/eltwise_layer.$(OBJ) \
  cbin/euclidean_loss_layer.$(OBJ) cbin/flatten_layer.$(OBJ) cbin/hinge_loss_layer.$(OBJ) cbin/im2col_layer.$(OBJ)  \
  cbin/infogain_loss_layer.$(OBJ) cbin/inner_product_layer.$(OBJ) \
  cbin/loss_layer.$(OBJ) cbin/lrn_layer.$(OBJ) \
  cbin/memory_data_layer.$(OBJ) cbin/multinomial_logistic_loss_layer.$(OBJ) cbin/mvn_layer.$(OBJ) cbin/neuron_layer.$(OBJ) \
  cbin/pooling_layer.$(OBJ) cbin/power_layer.$(OBJ) cbin/relu_layer.$(OBJ) \
  cbin/sigmoid_cross_entropy_loss_layer.$(OBJ) cbin/sigmoid_layer.$(OBJ) \
  cbin/silence_layer.$(OBJ) cbin/slice_layer.$(OBJ) cbin/softmax_layer.$(OBJ) \
  cbin/softmax_loss_layer.$(OBJ) cbin/split_layer.$(OBJ) cbin/tanh_layer.$(OBJ) \
  cbin/threshold_layer.$(OBJ)

CUDA_OBJS=gbin/math_functions.$(OBJ) gbin/im2col.$(OBJ) \
  gbin/absval_layer.$(OBJ) gbin/bnll_layer.$(OBJ) \
  gbin/concat_layer.$(OBJ) gbin/contrastive_loss_layer.$(OBJ) gbin/conv_layer.$(OBJ) \
  gbin/cudnn_conv_layer.$(OBJ) gbin/cudnn_pooling_layer.$(OBJ) gbin/cudnn_relu_layer.$(OBJ) \
  gbin/cudnn_sigmoid_layer.$(OBJ) gbin/cudnn_softmax_layer.$(OBJ) gbin/cudnn_tanh_layer.$(OBJ) \
  gbin/dropout_layer.$(OBJ) gbin/eltwise_layer.$(OBJ) gbin/euclidean_loss_layer.$(OBJ) \
  gbin/flatten_layer.$(OBJ) gbin/im2col_layer.$(OBJ) gbin/inner_product_layer.$(OBJ) gbin/lrn_layer.$(OBJ) \
  gbin/mvn_layer.$(OBJ) gbin/pooling_layer.$(OBJ) gbin/power_layer.$(OBJ) \
  gbin/relu_layer.$(OBJ) gbin/sigmoid_cross_entropy_loss_layer.$(OBJ) gbin/sigmoid_layer.$(OBJ) \
  gbin/silence_layer.$(OBJ) gbin/slice_layer.$(OBJ) gbin/softmax_layer.$(OBJ) \
  gbin/softmax_loss_layer.$(OBJ) gbin/split_layer.$(OBJ) gbin/tanh_layer.$(OBJ) \
  gbin/threshold_layer.$(OBJ) 

.SUFFIXES: .$(OBJ) .c .cpp .cu

all: cbin gbin $(LIBPREPEND)caffe$(LIBAPPEND) 

# build the library
$(LIBPREPEND)caffe$(LIBAPPEND): $(CPU_OBJS) $(CUDA_OBJS)
	$(GLD) $(LDFLAGS) $(CPU_OBJS) $(CUDA_OBJS) $(CUDA_LIBS) $(MKL_LIBS) $(GOOGLE_LIBS) $(OUTFLG)$@

# compile layer CPU code
cbin/%.$(OBJ) : layers/%.cpp
	$(CPP) $(CPPFLAGS) layers/$*.cpp $(CPPOUTFLG)cbin/$*.$(OBJ)

# layers that depend on protobuf headers
cbin/base_data_layer.$(OBJ) cbin/data_layer.$(OBJ) : proto/caffe.pb.h proto/caffe_pretty_print.pb.h

# compile CPU utils
cbin/%.$(OBJ) : util/%.cpp
	$(CPP) $(CPPFLAGS) util/$*.cpp $(CPPOUTFLG)cbin/$*.$(OBJ)

# util files that depend on protobuf headers
cbin/io.$(OBJ) cbin/upgrade_proto.$(OBJ) : proto/caffe.pb.h proto/caffe_pretty_print.pb.h

# compile protobuf code
cbin/%.$(OBJ) : proto/%.cc proto/caffe.pb.h proto/caffe_pretty_print.pb.h
	$(CPP) $(CPPFLAGS) proto/$*.cc $(CPPOUTFLG)cbin/$*.$(OBJ)

# compile main directory functions
cbin/%.$(OBJ) : %.cpp proto/caffe.pb.h proto/caffe_pretty_print.pb.h
	$(CPP) $(CPPFLAGS) $*.cpp $(CPPOUTFLG)cbin/$*.$(OBJ)

# compile GPU layers
gbin/%.$(OBJ) : layers/%.cu
	$(NVCC) $(NVCCFLAGS) layers/$*.cu $(NVCCOUTFLG)gbin/$*.$(OBJ)

# compile GPU utils
gbin/%.$(OBJ) : util/%.cu
	$(NVCC) $(NVCCFLAGS) util/$*.cu $(NVCCOUTFLG)gbin/$*.$(OBJ)

# make directories if needed
cbin :
	mkdir -p cbin

gbin :
	mkdir -p gbin

# run protoc on main proto
proto/caffe.pb.cc proto/caffe.pb.h : proto/caffe.proto
	protoc --cpp_out=proto --proto_path=proto proto/caffe.proto
	mkdir -p ../../include/caffe/proto
	cp proto/caffe.pb.h ../../include/caffe/proto/caffe.pb.h

# run protoc on pretty print proto
proto/caffe_pretty_print.pb.cc proto/caffe_pretty_print.pb.h: proto/caffe_pretty_print.proto
	protoc --cpp_out=proto --proto_path=proto proto/caffe_pretty_print.proto
	mkdir -p ../../include/caffe/proto
	cp proto/caffe_pretty_print.pb.h ../../include/caffe/proto/caffe_pretty_print.pb.h

install: all
	cp $(LIBPREPEND)caffe$(LIBAPPEND) $(INSTALL_DIR)

clean:
	rm -f *.$(OBJ) cbin/*.$(OBJ) gbin/*.$(OBJ) proto/*.cc proto/*.h ../../include/caffe/proto/*.h *$(LIBAPPEND) *.pdb *.exp *.lib

distclean: clean
	rm -f  *.jnilib Makefile.incl

