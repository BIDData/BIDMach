#include <jni.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/data_layers.hpp"

using namespace caffe;

using boost::shared_ptr;

static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

extern "C" {

// Static CAFFE class for setting global state

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_CAFFE_set_1mode
(JNIEnv * env, jobject calling_obj, int mode) { 
  if (mode == 0) {
    Caffe::set_mode(Caffe::CPU); 
  } else {
    Caffe::set_mode(Caffe::GPU); 
  }
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_CAFFE_init
(JNIEnv * env, jobject calling_obj, jint logtostderr, jint stderrthreshold, jint minloglevel, jstring jlogdir) { 
  char str[] = "bidmach_caffe";   
  char *pstr = new char[strlen(str)+1];
  strcpy(pstr, str);
  const char* logdir = (const char *)(env->GetStringUTFChars(jlogdir, 0));
  char* plogdir = new char[strlen(logdir)+1];
  strcpy(plogdir, logdir);
  FLAGS_log_dir = plogdir;
  FLAGS_logtostderr = logtostderr;
  FLAGS_stderrthreshold = stderrthreshold;
  FLAGS_minloglevel = minloglevel;
  ::google::InitGoogleLogging(pstr);
  env->ReleaseStringUTFChars(jlogdir, logdir);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_CAFFE_set_1phase
(JNIEnv * env, jobject calling_obj, int phase) { 
  if (phase == 0) {
    Caffe::set_phase(Caffe::TRAIN); 
  } else {
    Caffe::set_phase(Caffe::TEST);
  }
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_CAFFE_get_1mode
(JNIEnv * env, jobject calling_obj) { 
  if (Caffe::mode() == Caffe::CPU) {
    return 0;
  } else {
    return 1;
  }
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_CAFFE_get_1phase
(JNIEnv * env, jobject calling_obj) { 
  if (Caffe::phase() == Caffe::TRAIN) {
    return 0;
  } else {
    return 1;
  }
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_CAFFE_set_1device
(JNIEnv * env, jobject calling_obj, jint device_id) { 
  Caffe::SetDevice(device_id); 
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_CAFFE_DeviceQuery
(JNIEnv * env, jobject calling_obj) { 
  Caffe::DeviceQuery(); 
}

// NET class methods for managing networks

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_NET_netFromParamFile
(JNIEnv * env, jobject calling_obj, jstring jparamfile) {
  const char *paramfile = (const char *)(env->GetStringUTFChars(jparamfile, 0));
  jlong retv = 0;
  try {
    CheckFile(paramfile);
    Net<float> *net = new Net<float>(paramfile);
    retv = (jlong)(new shared_ptr<Net<float> >(net));
  } catch (std::exception e) {
    std::cerr << e.what();
  }
  env->ReleaseStringUTFChars(jparamfile, paramfile);
  return retv;
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_NET_netFromPretrained
(JNIEnv * env, jobject calling_obj, jstring jparamfile, jstring jpretrained) {
  const char *paramfile = (const char *)(env->GetStringUTFChars(jparamfile, 0));
  const char *pretrained = (const char *)(env->GetStringUTFChars(jpretrained, 0));
  CheckFile(paramfile);
  CheckFile(pretrained);
  Net<float> *net = new Net<float>(paramfile);
  net->CopyTrainedLayersFrom(pretrained);
  env->ReleaseStringUTFChars(jpretrained, pretrained);
  env->ReleaseStringUTFChars(jparamfile, paramfile);
  return (jlong)(new shared_ptr<Net<float> >(net));
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_NET_num_1inputs
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return net->num_inputs();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_NET_num_1outputs
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return net->num_outputs();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_NET_num_1layers
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return net->layers().size();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_NET_num_1blobs
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return net->blobs().size();
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_NET_layer
(JNIEnv * env, jobject calling_obj, jlong netRef, jint i) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return (jlong)(new shared_ptr<Layer<float> >(net->layers().at(i)));
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_NET_blob
(JNIEnv * env, jobject calling_obj, jlong netRef, jint i) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return (jlong)(new shared_ptr<Blob<float> >(net->blobs().at(i)));
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_NET_input_1blob
(JNIEnv * env, jobject calling_obj, jlong netRef, jint i) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return net->input_blob_indices().at(i);
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_NET_output_1blob
(JNIEnv * env, jobject calling_obj, jlong netRef, jint i) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  return net->output_blob_indices().at(i);
}

JNIEXPORT jobjectArray JNICALL Java_edu_berkeley_bvlc_NET_blob_1names
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  int size = net->blobs().size();
  jclass stringCls = env->FindClass("Ljava/lang/String;");
  jobjectArray result = env->NewObjectArray(size, stringCls, NULL);
  if (result == NULL) return NULL;
  for (int i = 0; i < size; i++) {
    env->SetObjectArrayElement(result, i, env->NewStringUTF(net->blob_names().at(i).c_str())); 
  }
  return result;
}

JNIEXPORT jobjectArray JNICALL Java_edu_berkeley_bvlc_NET_layer_1names
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  int size = net->layers().size();
  jclass stringCls = env->FindClass("Ljava/lang/String;");
  jobjectArray result = env->NewObjectArray(size, stringCls, NULL);
  if (result == NULL) return NULL;
  for (int i = 0; i < size; i++) {
    env->SetObjectArrayElement(result, i, env->NewStringUTF(net->layer_names().at(i).c_str())); 
  }
  return result;
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_NET_forward
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  net->ForwardPrefilled();
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_NET_backward
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  net->Backward();
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_NET_blob_1by_1name
(JNIEnv * env, jobject calling_obj, jlong netRef, jstring jbname) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  const char *bname = (const char *)(env->GetStringUTFChars(jbname, 0));
  jlong ref = (jlong)(new shared_ptr<Blob<float> >(net->blob_by_name(bname)));
  env->ReleaseStringUTFChars(jbname, bname);
  return ref;
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_NET_layer_1by_1name
(JNIEnv * env, jobject calling_obj, jlong netRef, jstring jbname) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  const char *bname = (const char *)(env->GetStringUTFChars(jbname, 0));
  jlong ref = (jlong)(new shared_ptr<Layer<float> >(net->layer_by_name(bname)));
  env->ReleaseStringUTFChars(jbname, bname);
  return ref;
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_NET_clearNet
(JNIEnv * env, jobject calling_obj, jlong addr) { 
  delete (shared_ptr<Net<float> > *)addr;
}


JNIEXPORT jobjectArray JNICALL Java_edu_berkeley_bvlc_NET_bottom_1ids
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  int size = net->layers().size();
  vector<vector<int> > idvecs = net->bottom_id_vecs();
  jclass intarrayCls = env->FindClass("[I");
  jobjectArray result = env->NewObjectArray(size, intarrayCls, NULL);
  for (int i = 0; i < size; i++) {
    int sz = idvecs[i].size();
    jintArray vec = NULL;
    if (sz > 0) {
      vec = env->NewIntArray(sz);
      env->SetIntArrayRegion(vec, 0, sz, &(idvecs[i][0]));
    }
    env->SetObjectArrayElement(result, i, vec); 
  }
  return result;
}

JNIEXPORT jobjectArray JNICALL Java_edu_berkeley_bvlc_NET_top_1ids
(JNIEnv * env, jobject calling_obj, jlong netRef) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  int size = net->layers().size();
  vector<vector<int> > idvecs = net->top_id_vecs();
  jclass intarrayCls = env->FindClass("[I");
  jobjectArray result = env->NewObjectArray(size, intarrayCls, NULL);
  for (int i = 0; i < size; i++) {
    int sz = idvecs[i].size();
    jintArray vec = NULL;
    if (sz > 0) {
      vec = env->NewIntArray(sz);
      env->SetIntArrayRegion(vec, 0, sz, &(idvecs[i][0]));
    }
    env->SetObjectArrayElement(result, i, vec); 
  }
  return result;
}

// BLOB class methods

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_BLOB_count
(JNIEnv * env, jobject calling_obj, jlong blobRef) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  return blob->count();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_BLOB_num
(JNIEnv * env, jobject calling_obj, jlong blobRef) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  return blob->num();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_BLOB_channels
(JNIEnv * env, jobject calling_obj, jlong blobRef) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  return blob->channels();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_BLOB_width
(JNIEnv * env, jobject calling_obj, jlong blobRef) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  return blob->width();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_BLOB_height
(JNIEnv * env, jobject calling_obj, jlong blobRef) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  return blob->height();
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_BLOB_offset
(JNIEnv * env, jobject calling_obj, jlong blobRef, jint n, jint c, jint h, jint w) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  return blob->offset(n, c, h, w);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_BLOB_get_1data
(JNIEnv * env, jobject calling_obj, jlong blobRef, jfloatArray ja, jint n) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  jfloat *a = (jfloat *)(env->GetPrimitiveArrayCritical(ja, JNI_FALSE));
  switch (Caffe::mode()) {
  case Caffe::GPU:
    cudaMemcpy(a, blob->gpu_data(), n*sizeof(float), cudaMemcpyDeviceToHost);
    break;
  case Caffe::CPU:
    memcpy(a, blob->cpu_data(), n*sizeof(float));
    break;
  default:
    break;
  }
  env->ReleasePrimitiveArrayCritical(ja, a, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_BLOB_put_1data
(JNIEnv * env, jobject calling_obj, jlong blobRef, jfloatArray ja, jint n) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  jfloat *a = (jfloat *)(env->GetPrimitiveArrayCritical(ja, JNI_FALSE));
  switch (Caffe::mode()) {
  case Caffe::GPU:
    cudaMemcpy(blob->mutable_gpu_data(), a, n*sizeof(float), cudaMemcpyHostToDevice);
    break;
  case Caffe::CPU:
    memcpy(blob->mutable_cpu_data(), a, n*sizeof(float));
    break;
  default:
    break;
  }
  env->ReleasePrimitiveArrayCritical(ja, a, 0);
}


JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_BLOB_get_1diff
(JNIEnv * env, jobject calling_obj, jlong blobRef, jfloatArray ja, jint n) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  jfloat *a = (jfloat *)(env->GetPrimitiveArrayCritical(ja, JNI_FALSE));
  switch (Caffe::mode()) {
  case Caffe::GPU:
    cudaMemcpy(a, blob->gpu_diff(), n*sizeof(float), cudaMemcpyDeviceToHost);
    break;
  case Caffe::CPU:
    memcpy(a, blob->cpu_diff(), n*sizeof(float));
    break;
  default:
    break;
  }
  env->ReleasePrimitiveArrayCritical(ja, a, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_BLOB_put_1diff
(JNIEnv * env, jobject calling_obj, jlong blobRef, jfloatArray ja, jint n) {
  shared_ptr<Blob<float> > blob = *((shared_ptr<Blob<float> > *)blobRef);
  jfloat *a = (jfloat *)(env->GetPrimitiveArrayCritical(ja, JNI_FALSE));
  switch (Caffe::mode()) {
  case Caffe::GPU:
    cudaMemcpy(blob->mutable_gpu_diff(), a, n*sizeof(float), cudaMemcpyHostToDevice);
    break;
  case Caffe::CPU:
    memcpy(blob->mutable_cpu_diff(), a, n*sizeof(float));
    break;
  default:
    break;
  }
  env->ReleasePrimitiveArrayCritical(ja, a, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_BLOB_clearBlob
(JNIEnv * env, jobject calling_obj, jlong addr) { 
  delete (shared_ptr<Blob<float> > *)addr;
}

// LAYER class methods

JNIEXPORT jint JNICALL Java_edu_berkeley_bvlc_LAYER_num_1blobs
(JNIEnv * env, jobject calling_obj, jlong layerRef) {
  shared_ptr<Layer<float> > layer = *((shared_ptr<Layer<float> > *)layerRef);
  return layer->blobs().size();
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_LAYER_blob
(JNIEnv * env, jobject calling_obj, jlong layerRef, jint i) {
  shared_ptr<Layer<float> > layer = *((shared_ptr<Layer<float> > *)layerRef);
  return (jlong)(new shared_ptr<Blob<float> >(layer->blobs().at(i)));
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_LAYER_clearLayer
(JNIEnv * env, jobject calling_obj, jlong addr) { 
  delete (shared_ptr<Layer<float> > *)addr;
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_LAYER_pushMemoryData
(JNIEnv * env, jobject calling_obj, jlong layerRef, jfloatArray jA, jfloatArray jB, 
 int num, int nchannels, int height, int width) {
  shared_ptr<MemoryDataLayer<float> > layer = *((shared_ptr<MemoryDataLayer<float> > *)layerRef);
  float *A = (jfloat *)(env->GetPrimitiveArrayCritical(jA, JNI_FALSE));
  float *B = (jfloat *)(env->GetPrimitiveArrayCritical(jB, JNI_FALSE));
  layer->AddData(A, B, num, nchannels, height, width);
  env->ReleasePrimitiveArrayCritical(jB, B, 0);
  env->ReleasePrimitiveArrayCritical(jA, A, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_LAYER_forwardFromTo
(JNIEnv * env, jobject calling_obj, jlong netRef, jint from, jint to) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  net->ForwardFromTo(from, to);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_LAYER_backwardFromTo
(JNIEnv * env, jobject calling_obj, jlong netRef, jint from, jint to) {
  shared_ptr<Net<float> > net = *((shared_ptr<Net<float> > *)netRef);
  net->BackwardFromTo(from, to);
}

  /*JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_LAYER_im2col
(JNIEnv * env, jobject calling_obj, jfloatArray ja, jint nchannels, jint height, jint width, jint ksize,
 jint pad, jint stride, jfloatArray jb) {
  jfloat *a = (jfloat *)(env->GetPrimitiveArrayCritical(ja, JNI_FALSE));
  jfloat *b = (jfloat *)(env->GetPrimitiveArrayCritical(jb, JNI_FALSE));

  im2col_gpu(a, channels, height, width, ksize, pad, stride, b);

  env->ReleasePrimitiveArrayCritical(jb, b, 0);
  env->ReleasePrimitiveArrayCritical(ja, a, 0);
  }*/

//
// SGDSOLVER params
//

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_SGDSOLVER_fromParams
(JNIEnv * env, jobject calling_obj, jstring jparamfile) {
  const char *paramfile = (const char *)(env->GetStringUTFChars(jparamfile, 0));
  CheckFile(paramfile);
  SGDSolver<float> *solver = new SGDSolver<float>(paramfile);
  env->ReleaseStringUTFChars(jparamfile, paramfile);
  return (jlong)(new shared_ptr<SGDSolver<float> >(solver));
}

JNIEXPORT jlong JNICALL Java_edu_berkeley_bvlc_SGDSOLVER_net
(JNIEnv * env, jobject calling_obj, jlong solverRef) {
  shared_ptr<SGDSolver<float> > solver = *((shared_ptr<SGDSolver<float> > *)solverRef);
  return (jlong)(new shared_ptr<Net<float> >(solver->net()));
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_SGDSOLVER_Solve
(JNIEnv * env, jobject calling_obj, jlong solverRef) {
  shared_ptr<SGDSolver<float> > solver = *((shared_ptr<SGDSolver<float> > *)solverRef);
  solver->Solve();
}

JNIEXPORT void JNICALL Java_edu_berkeley_bvlc_SGDSOLVER_SolveResume
(JNIEnv * env, jobject calling_obj, jlong solverRef, jstring jsavefile) {
  const char *savefile = (const char *)(env->GetStringUTFChars(jsavefile, 0));
  CheckFile(savefile);
  shared_ptr<SGDSolver<float> > solver = *((shared_ptr<SGDSolver<float> > *)solverRef);
  solver->Solve(savefile);
  env->ReleaseStringUTFChars(jsavefile, savefile);
}

}
