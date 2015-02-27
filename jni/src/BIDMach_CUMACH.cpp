
#include <jni.h>
#include <cuda_runtime.h>
#include <Logger.hpp>
#include <JNIUtils.hpp>
#include <PointerUtils.hpp>
#include <MatKernel.hpp>


extern "C" {

  JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
  {
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
      {
        return JNI_ERR;
      }

    Logger::log(LOG_TRACE, "Initializing JCumat\n");

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    return JNI_VERSION_1_4;

  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_LDAgibbs
  (JNIEnv *env, jobject obj, jint nrows, jint nnz, jobject jA, jobject jB, jobject jAN, jobject jBN, 
   jobject jCir, jobject jCic, jobject jP, jfloat nsamps)
  {
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *AN = (float*)getPointer(env, jAN);
    float *BN = (float*)getPointer(env, jBN);
    int *Cir = (int*)getPointer(env, jCir);
    int *Cic = (int*)getPointer(env, jCic);
    float *P = (float*)getPointer(env, jP);

    return LDA_Gibbs(nrows, nnz, A, B, AN, BN, Cir, Cic, P, nsamps);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_LDAgibbsx
  (JNIEnv *env, jobject obj, jint nrows, jint nnz, jobject jA, jobject jB,
   jobject jCir, jobject jCic, jobject jP, jobject jMs, jobject jUs, int k)
  {
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    int *Cir = (int*)getPointer(env, jCir);
    int *Cic = (int*)getPointer(env, jCic);
    float *P = (float*)getPointer(env, jP);
    int *Ms = (int*)getPointer(env, jMs);
    int *Us = (int*)getPointer(env, jUs);

    return LDA_Gibbs1(nrows, nnz, A, B, Cir, Cic, P, Ms, Us, k);
  }
  
  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_LDAgibbsv
  (JNIEnv *env, jobject obj, jint nrows, jint nnz, jobject jA, jobject jB, jobject jAN, jobject jBN, 
   jobject jCir, jobject jCic, jobject jP, jobject jnsamps)
  // (JNIEnv *env, jobject obj, jint nrows, jint nnz, jobject jA, jobject jB, jobject jAN, jobject jBN, 
  // jobject jCir, jobject jCic, jobject jP, jfloat nsamps)
  {
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *AN = (float*)getPointer(env, jAN);
    float *BN = (float*)getPointer(env, jBN);
    int *Cir = (int*)getPointer(env, jCir);
    int *Cic = (int*)getPointer(env, jCic);
    float *P = (float*)getPointer(env, jP);
    float *nsamps = (float*)getPointer(env, jnsamps);
    return LDA_Gibbsv(nrows, nnz, A, B, AN, BN, Cir, Cic, P, nsamps);
    //return LDA_Gibbs(nrows, nnz, A, B, AN, BN, Cir, Cic, P, nsamps);
  }
  

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_veccmp
  (JNIEnv *env, jobject obj, jobject ja, jobject jb, jobject jc)
  {
    int *a = (int*)getPointer(env, ja);
    int *b = (int*)getPointer(env, jb);
    int *c = (int*)getPointer(env, jc);

    return veccmp(a, b, c);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_hammingdists
  (JNIEnv *env, jobject obj, jobject ja, jobject jb, jobject jw, jobject jop, jobject jow, jint n)
  {
    int *a = (int*)getPointer(env, ja);
    int *b = (int*)getPointer(env, jb);
    int *w = (int*)getPointer(env, jw);
    int *op = (int*)getPointer(env, jop);
    int *ow = (int*)getPointer(env, jow);

    return hammingdists(a, b, w, op, ow, n);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applylinks
  (JNIEnv *env, jobject obj, jobject jA, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    float *nativeA = (float*)getPointer(env, jA);
    int *nativeL = (int*)getPointer(env, jL);
    float *nativeC = (float*)getPointer(env, jC);

    return apply_links(nativeA, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applypreds
  (JNIEnv *env, jobject obj, jobject jA, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    float *nativeA = (float*)getPointer(env, jA);
    int *nativeL = (int*)getPointer(env, jL);
    float *nativeC = (float*)getPointer(env, jC);

    return apply_preds(nativeA, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applylls
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    float *nativeA = (float*)getPointer(env, jA);
    float *nativeB = (float*)getPointer(env, jB);
    int *nativeL = (int*)getPointer(env, jL);
    float *nativeC = (float*)getPointer(env, jC);

    return apply_lls(nativeA, nativeB, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applyderivs
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    float *nativeA = (float*)getPointer(env, jA);
    float *nativeB = (float*)getPointer(env, jB);
    int *nativeL = (int*)getPointer(env, jL);
    float *nativeC = (float*)getPointer(env, jC);

    return apply_derivs(nativeA, nativeB, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_treePack
  (JNIEnv *env, jobject obj, jobject jfdata, jobject jtreenodes, jobject jicats, jobject jout, jobject jfieldlens, jint nrows, jint ncols, jint ntrees, jint nsamps, jint seed) 
  {
    float *fdata = (float*)getPointer(env, jfdata);
    int *treenodes = (int*)getPointer(env, jtreenodes);
    int *icats = (int*)getPointer(env, jicats);
    long long *out = (long long*)getPointer(env, jout);
    int *fieldlens = (int*)getPointer(env, jfieldlens);

    return treePack(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_treePackfc
  (JNIEnv *env, jobject obj, jobject jfdata, jobject jtreenodes, jobject jfcats, jobject jout, jobject jfieldlens, jint nrows, jint ncols, jint ntrees, jint nsamps, jint seed) 
  {
    float *fdata = (float*)getPointer(env, jfdata);
    int *treenodes = (int*)getPointer(env, jtreenodes);
    float *fcats = (float*)getPointer(env, jfcats);
    long long *out = (long long*)getPointer(env, jout);
    int *fieldlens = (int*)getPointer(env, jfieldlens);

    return treePackfc(fdata, treenodes, fcats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_treePackInt
  (JNIEnv *env, jobject obj, jobject jfdata, jobject jtreenodes, jobject jicats, jobject jout, jobject jfieldlens, jint nrows, jint ncols, jint ntrees, jint nsamps, jint seed) 
  {
    int *fdata = (int*)getPointer(env, jfdata);
    int *treenodes = (int*)getPointer(env, jtreenodes);
    int *icats = (int*)getPointer(env, jicats);
    long long *out = (long long*)getPointer(env, jout);
    int *fieldlens = (int*)getPointer(env, jfieldlens);

    return treePackInt(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  }

 JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_minImpurity
 (JNIEnv *env, jobject obj, jobject jkeys, jobject jcounts, jobject joutv, jobject joutf, jobject joutg, jobject joutc, jobject jjc, jobject jfieldlens, 
   jint nnodes, jint ncats, jint nsamps, jint impType) 
  {
    long long *keys = (long long*)getPointer(env, jkeys);
    int *counts = (int*)getPointer(env, jcounts);
    int *outv = (int*)getPointer(env, joutv);
    int *outf = (int*)getPointer(env, joutf);
    float *outg = (float*)getPointer(env, joutg);
    int *outc = (int*)getPointer(env, joutc);
    int *jc = (int*)getPointer(env, jjc);
    int *fieldlens = (int*)getPointer(env, jfieldlens);

    return minImpurity(keys, counts, outv, outf, outg, outc, jc, fieldlens, nnodes, ncats, nsamps, impType);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_findBoundaries
  (JNIEnv *env, jobject obj, jobject jkeys, jobject jjc, jint n, jint njc, jint shift)
  {
    long long *keys = (long long*)getPointer(env, jkeys);
    int *jc = (int*)getPointer(env, jjc);

    return findBoundaries(keys, jc, n, njc, shift);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_treeWalk
  (JNIEnv *env, jobject obj, jobject jfdata, jobject jinodes, jobject jfnodes, jobject jitrees, jobject jftrees, jobject jvtrees, jobject jctrees,
   jint nrows, jint ncols, jint ntrees, jint nnodes, jint getcat, jint nbits, jint nlevels)
  {
    float *fdata = (float*)getPointer(env, jfdata);
    int *inodes = (int*)getPointer(env, jinodes);
    float *fnodes = (float*)getPointer(env, jfnodes);
    int *itrees = (int*)getPointer(env, jitrees);
    int *ftrees = (int*)getPointer(env, jftrees);
    int *vtrees = (int*)getPointer(env, jvtrees);
    float *ctrees = (float*)getPointer(env, jctrees);

    return treeWalk(fdata, inodes, fnodes, itrees, ftrees, vtrees, ctrees, nrows, ncols, ntrees, nnodes, getcat, nbits, nlevels);  
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_floatToInt
  (JNIEnv *env, jobject obj, jint n, jobject jin, jobject jout, jint nbits)
  {
    float *in = (float*)getPointer(env, jin);
    int *out = (int*)getPointer(env, jout);

    return floatToInt(n, in, out, nbits);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_jfeatsToIfeats
  (JNIEnv *env, jobject obj, jint itree, jobject jinodes, jobject jjfeats, jobject jifeats, jint n, jint nfeats, jint seed)
  {
    int *inodes = (int*)getPointer(env, jinodes);
    int *jfeats = (int*)getPointer(env, jjfeats);
    int *ifeats = (int*)getPointer(env, jifeats);

    return jfeatsToIfeats(itree, inodes, jfeats, ifeats, n, nfeats, seed);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_hashMult
  (JNIEnv *env, jobject obj, jint nrows, jint nfeats, jint ncols, 
   jobject jA, jobject jBdata, jobject jBir, jobject jBjc, jobject jC, jint transpose)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bjc = (int*)getPointer(env, jBjc);
    float *C = (float*)getPointer(env, jC);

    return hashmult(nrows, nfeats, ncols, A, Bdata, Bir, Bjc, C, transpose);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_hashCross
  (JNIEnv *env, jobject obj, jint nrows, jint nfeats, jint ncols, jobject jA,
   jobject jBdata, jobject jBir, jobject jBjc,
   jobject jCdata, jobject jCir, jobject jCjc, jobject jD, jint transpose)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bjc = (int*)getPointer(env, jBjc);
    float *Cdata = (float*)getPointer(env, jCdata);
    int *Cir = (int*)getPointer(env, jCir);
    int *Cjc = (int*)getPointer(env, jCjc);
    float *D = (float*)getPointer(env, jD);

    return hashcross(nrows, nfeats, ncols, A, Bdata, Bir, Bjc, Cdata, Cir, Cjc, D, transpose);
  }

}
