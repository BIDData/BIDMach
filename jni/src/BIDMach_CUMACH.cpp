
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

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_LDAgibbsBino
  (JNIEnv *env, jobject obj, jint nrows, jint nnz, jobject jA, jobject jB, jobject jAN, jobject jBN, 
   jobject jCir, jobject jCic, jobject jCv, jobject jP, jint nsamps)
  {
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *AN = (float*)getPointer(env, jAN);
    float *BN = (float*)getPointer(env, jBN);
    int *Cir = (int*)getPointer(env, jCir);
    int *Cic = (int*)getPointer(env, jCic);
    float *Cv = (float*)getPointer(env, jCv);
    float *P = (float*)getPointer(env, jP);

    return LDA_GibbsBino(nrows, nnz, A, B, AN, BN, Cir, Cic, Cv, P, nsamps);
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

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applydlinks
  (JNIEnv *env, jobject obj, jobject jA, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    double *nativeA = (double*)getPointer(env, jA);
    int *nativeL = (int*)getPointer(env, jL);
    double *nativeC = (double*)getPointer(env, jC);

    return apply_dlinks(nativeA, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applydpreds
  (JNIEnv *env, jobject obj, jobject jA, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    double *nativeA = (double*)getPointer(env, jA);
    int *nativeL = (int*)getPointer(env, jL);
    double *nativeC = (double*)getPointer(env, jC);

    return apply_dpreds(nativeA, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applydlls
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    double *nativeA = (double*)getPointer(env, jA);
    double *nativeB = (double*)getPointer(env, jB);
    int *nativeL = (int*)getPointer(env, jL);
    double *nativeC = (double*)getPointer(env, jC);

    return apply_dlls(nativeA, nativeB, nativeL, nativeC, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applydderivs
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jobject jL, jobject jC, jint nrows, jint ncols) 
  {
    double *nativeA = (double*)getPointer(env, jA);
    double *nativeB = (double*)getPointer(env, jB);
    int *nativeL = (int*)getPointer(env, jL);
    double *nativeC = (double*)getPointer(env, jC);

    return apply_dderivs(nativeA, nativeB, nativeL, nativeC, nrows, ncols);
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
  (JNIEnv *env, jobject obj, jint nrows, jint nfeats, jint ncols, jint brows1, jint brows2,
   jobject jA, jobject jBdata, jobject jBir, jobject jBjc, jobject jC, jint transpose)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bjc = (int*)getPointer(env, jBjc);
    float *C = (float*)getPointer(env, jC);

    return hashmult(nrows, nfeats, ncols, brows1, brows2, A, Bdata, Bir, Bjc, C, transpose);
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

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_multinomial
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jobject jA, jobject jB, jobject jNorm, jint nvals)
  {
    float *A = (float*)getPointer(env, jA);
    int *B = (int*)getPointer(env, jB);
    float *Norm = (float*)getPointer(env, jNorm);

    return multinomial(nrows, ncols, A, B, Norm, nvals);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_multinomial2
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jobject jA, jobject jB, jint nvals)
  {
    float *A = (float*)getPointer(env, jA);
    int *B = (int*)getPointer(env, jB);

    return multinomial2(nrows, ncols, A, B, nvals);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_multADAGrad
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, int nnz, jobject jA, jobject jBdata, jobject jBir, jobject jBic,
   jobject jMM, jobject jSumsq, jobject jMask, int maskrows, jobject jlrate, jint lrlen, jobject jvexp, jint vexplen,
   jobject jtexp, jint texplen, float istep, jint addgrad, float epsilon, jint biasv, jint nbr)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bic = (int*)getPointer(env, jBic);
    float *MM = (float*)getPointer(env, jMM);
    float *Sumsq = (float*)getPointer(env, jSumsq);
    float *Mask = (float*)getPointer(env, jMask);
    float *lrate = (float*)getPointer(env, jlrate);
    float *vexp = (float*)getPointer(env, jvexp);
    float *texp = (float*)getPointer(env, jtexp);

    return multADAGrad(nrows, ncols, nnz, A, Bdata, Bir, Bic, MM, Sumsq, Mask, maskrows, lrate, lrlen,
                       vexp, vexplen, texp, texplen, istep, addgrad, epsilon, biasv, nbr);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_multADAGradTile
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint y, jint x, int nnz, jobject jA, jint lda, jobject jBdata, jobject jBir, jobject jBic,
   jobject jMM, jobject jSumsq, jobject jMask, int maskrows, jobject jlrate, jint lrlen, jobject jvexp, jint vexplen,
   jobject jtexp, jint texplen, float istep, jint addgrad, float epsilon, jint biasv, jint nbr)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bic = (int*)getPointer(env, jBic);
    float *MM = (float*)getPointer(env, jMM);
    float *Sumsq = (float*)getPointer(env, jSumsq);
    float *Mask = (float*)getPointer(env, jMask);
    float *lrate = (float*)getPointer(env, jlrate);
    float *vexp = (float*)getPointer(env, jvexp);
    float *texp = (float*)getPointer(env, jtexp);

    return multADAGradTile(nrows, ncols, y, x, nnz, A, lda, Bdata, Bir, Bic, MM, Sumsq, Mask, maskrows, lrate, lrlen,
                           vexp, vexplen, texp, texplen, istep, addgrad, epsilon, biasv, nbr);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_multGradTile
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint y, jint x, int nnz, jobject jA, jint lda, jobject jBdata, jobject jBir, jobject jBic,
   jobject jMM, jobject jMask, int maskrows, jobject jlrate, jint lrlen, jfloat limit, jint biasv, jint nbr)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bic = (int*)getPointer(env, jBic);
    float *MM = (float*)getPointer(env, jMM);
    float *Mask = (float*)getPointer(env, jMask);
    float *lrate = (float*)getPointer(env, jlrate);

    return multGradTile(nrows, ncols, y, x, nnz, A, lda, Bdata, Bir, Bic, MM, Mask, maskrows, lrate, lrlen, limit, biasv, nbr);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_hashmultADAGrad
  (JNIEnv *env, jobject obj, jint nrows, jint nfeats, jint ncols, jint brows1, jint brows2, jobject jA, jobject jBdata, jobject jBir, jobject jBjc, jint transpose,
   jobject jMM, jobject jSumsq, jobject jMask, int maskrows, jobject jlrate, jint lrlen, jobject jvexp, jint vexplen,
   jobject jtexp, jint texplen, float istep, jint addgrad, float epsilon)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bjc = (int*)getPointer(env, jBjc);
    float *MM = (float*)getPointer(env, jMM);
    float *Sumsq = (float*)getPointer(env, jSumsq);
    float *Mask = (float*)getPointer(env, jMask);
    float *lrate = (float*)getPointer(env, jlrate);
    float *vexp = (float*)getPointer(env, jvexp);
    float *texp = (float*)getPointer(env, jtexp);

    return hashmultADAGrad(nrows, nfeats, ncols, brows1, brows2, A, Bdata, Bir, Bjc, transpose,
                           MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
  }
  /*
  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecBlock
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwords, jint shift, jint npos, jint nneg, jobject jW, jobject jD, jobject jC, jfloat lrate)

  {
    float *C = (float*)getPointer(env, jC);
    float *D = (float*)getPointer(env, jD);
    int *W = (int*)getPointer(env, jW);

    return word2vecBlock(nrows, ncols, nwords, shift, npos, nneg, W, D, C, lrate);
  }
  */

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecPos
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint shift, jobject jW, jobject jLB, jobject jUB, jobject jA, jobject jB, jfloat lrate, jfloat vexp)

  {
    int *W = (int*)getPointer(env, jW);
    int *LB = (int*)getPointer(env, jLB);
    int *UB = (int*)getPointer(env, jUB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);

    return word2vecPos(nrows, ncols, shift, W, LB, UB, A, B, lrate, vexp);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecNeg
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jobject jWA, jobject jWB, jobject jA, jobject jB, jfloat lrate, jfloat vexp)
  {
    int *WA = (int*)getPointer(env, jWA);
    int *WB = (int*)getPointer(env, jWB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);

    return word2vecNeg(nrows, ncols, nwa, nwb, WA, WB, A, B, lrate, vexp);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecNegFilt
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwords, jint nwa, jint nwb, jobject jWA, jobject jWB, jobject jA, jobject jB, jfloat lrate, jfloat vexp)
  {
    int *WA = (int*)getPointer(env, jWA);
    int *WB = (int*)getPointer(env, jWB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);

    return word2vecNegFilt(nrows, ncols, nwords, nwa, nwb, WA, WB, A, B, lrate, vexp);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecEvalPos
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint shift, jobject jW, jobject jLB, jobject jUB, jobject jA, jobject jB, jobject jRetval)

  {
    int *W = (int*)getPointer(env, jW);
    int *LB = (int*)getPointer(env, jLB);
    int *UB = (int*)getPointer(env, jUB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *Retval = (float*)getPointer(env, jRetval);

    return word2vecEvalPos(nrows, ncols, shift, W, LB, UB, A, B, Retval);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecEvalNeg
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jobject jWA, jobject jWB, jobject jA, jobject jB, jobject jRetval)
  {
    int *WA = (int*)getPointer(env, jWA);
    int *WB = (int*)getPointer(env, jWB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *Retval = (float*)getPointer(env, jRetval);

    return word2vecEvalNeg(nrows, ncols, nwa, nwb, WA, WB, A, B, Retval);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecFwd
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jobject jWA, jobject jWB, jobject jA, jobject jB, jobject jC)
  {
    int *WA = (int*)getPointer(env, jWA);
    int *WB = (int*)getPointer(env, jWB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *C = (float*)getPointer(env, jC);

    return word2vecFwd(nrows, ncols, nwa, nwb, WA, WB, A, B, C);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_word2vecBwd
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jobject jWA, jobject jWB, jobject jA, jobject jB, jobject jC, jfloat lrate)
  {
    int *WA = (int*)getPointer(env, jWA);
    int *WB = (int*)getPointer(env, jWB);
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *C = (float*)getPointer(env, jC);

    return word2vecBwd(nrows, ncols, nwa, nwb, WA, WB, A, B, C, lrate);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applyfwd
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jint ifn, jint n)
  {
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);

    return apply_fwd(A, B, ifn, n);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_applyderiv
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jobject jC, jint ifn, jint n)
  {
    float *A = (float*)getPointer(env, jA);
    float *B = (float*)getPointer(env, jB);
    float *C = (float*)getPointer(env, jC);

    return apply_deriv(A, B, C, ifn, n);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_ADAGrad
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jobject jmm, jobject jum, jobject jssq, jobject jmask, jint maskr,
   jfloat nw, jobject jve, jint nve, jobject jts, jint nts, jobject jlr, jint nlr, jfloat langevin, jfloat eps, jint doupdate)
  {
    float *mm = (float*)getPointer(env, jmm);
    float *um = (float*)getPointer(env, jum);
    float *ssq = (float*)getPointer(env, jssq);
    float *mask = (float*)getPointer(env, jmask);
    float *ve = (float*)getPointer(env, jve);
    float *ts = (float*)getPointer(env, jts);
    float *lr = (float*)getPointer(env, jlr);

    return ADAGrad(nrows, ncols, mm, um, ssq, mask, maskr, nw, ve, nve, ts, nts, lr, nlr, langevin, eps, doupdate);
  }

    JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_ADAGradm
    (JNIEnv *env, jobject obj, jint nrows, jint ncols, jobject jmm, jobject jum, jobject jssq, jobject jmom, jfloat mu, jobject jmask, jint maskr,
     jfloat nw, jobject jve, jint nve, jobject jts, jint nts, jobject jlr, jint nlr, jfloat langevin, jfloat eps, jint doupdate)
  {
    float *mm = (float*)getPointer(env, jmm);
    float *um = (float*)getPointer(env, jum);
    float *ssq = (float*)getPointer(env, jssq);
    float *mom = (float*)getPointer(env, jmom);
    float *mask = (float*)getPointer(env, jmask);
    float *ve = (float*)getPointer(env, jve);
    float *ts = (float*)getPointer(env, jts);
    float *lr = (float*)getPointer(env, jlr);

    return ADAGradm(nrows, ncols, mm, um, ssq, mom, mu, mask, maskr, nw, ve, nve, ts, nts, lr, nlr, langevin, eps, doupdate);
  }
  
  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_ADAGradn
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jobject jmm, jobject jum, jobject jssq, jobject jmom, jfloat mu, jobject jmask, jint maskr,
   jfloat nw, jobject jve, jint nve, jobject jts, jint nts, jobject jlr, jint nlr, jfloat langevin, jfloat eps, jint doupdate)
  {
    float *mm = (float*)getPointer(env, jmm);
    float *um = (float*)getPointer(env, jum);
    float *ssq = (float*)getPointer(env, jssq);
    float *mom = (float*)getPointer(env, jmom);
    float *mask = (float*)getPointer(env, jmask);
    float *ve = (float*)getPointer(env, jve);
    float *ts = (float*)getPointer(env, jts);
    float *lr = (float*)getPointer(env, jlr);

    return ADAGradn(nrows, ncols, mm, um, ssq, mom, mu, mask, maskr, nw, ve, nve, ts, nts, lr, nlr, langevin, eps, doupdate);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_LSTMfwd
  (JNIEnv *env, jobject obj, jobject jinC, jobject jLIN1, jobject jLIN2, jobject jLIN3, jobject jLIN4, jobject joutC, jobject joutH, jint n)
  {
    float *inC = (float*)getPointer(env, jinC);
    float *LIN1 = (float*)getPointer(env, jLIN1);
    float *LIN2 = (float*)getPointer(env, jLIN2);
    float *LIN3 = (float*)getPointer(env, jLIN3);
    float *LIN4 = (float*)getPointer(env, jLIN4);
    float *outC = (float*)getPointer(env, joutC);
    float *outH = (float*)getPointer(env, joutH);

    return lstm_fwd(inC, LIN1, LIN2, LIN3, LIN4, outC, outH, n);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_LSTMbwd
  (JNIEnv *env, jobject obj, jobject jinC, jobject jLIN1, jobject jLIN2, jobject jLIN3, jobject jLIN4, jobject jdoutC, jobject jdoutH, 
   jobject jdinC, jobject jdLIN1, jobject jdLIN2, jobject jdLIN3, jobject jdLIN4, jint n)
  {
    float *inC = (float*)getPointer(env, jinC);
    float *LIN1 = (float*)getPointer(env, jLIN1);
    float *LIN2 = (float*)getPointer(env, jLIN2);
    float *LIN3 = (float*)getPointer(env, jLIN3);
    float *LIN4 = (float*)getPointer(env, jLIN4);
    float *doutC = (float*)getPointer(env, jdoutC);
    float *doutH = (float*)getPointer(env, jdoutH);
    float *dinC = (float*)getPointer(env, jdinC);
    float *dLIN1 = (float*)getPointer(env, jdLIN1);
    float *dLIN2 = (float*)getPointer(env, jdLIN2);
    float *dLIN3 = (float*)getPointer(env, jdLIN3);
    float *dLIN4 = (float*)getPointer(env, jdLIN4);

    return lstm_bwd(inC, LIN1, LIN2, LIN3, LIN4, doutC, doutH, dinC, dLIN1, dLIN2, dLIN3, dLIN4, n);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_pairembed
  (JNIEnv *env, jobject obj, jobject jA, jobject jB, jobject jC, jint n)
  {
    int *A = (int*)getPointer(env, jA);
    int *B = (int*)getPointer(env, jB);
    long long *C = (long long*)getPointer(env, jC);

    return pairembed(A, B, C, n);
  }

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_pairMultTile
  (JNIEnv *env, jobject obj, jint nrows, jint ncols, jint brows1, jint brows2, jobject jA, jint lda, jobject jA2, jint lda2,
   jobject jBdata, jobject jBir, jobject jBjc, jint broff, jint bcoff, jobject jC, jint ldc, jint transpose)
  {
    float *A = (float*)getPointer(env, jA);
    float *A2 = (float*)getPointer(env, jA2);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bjc = (int*)getPointer(env, jBjc);
    float *C = (float*)getPointer(env, jC);

    return pairMultTile(nrows, ncols, brows1, brows2, A, lda, A2, lda2, Bdata, Bir, Bjc, broff, bcoff, C, ldc, transpose);
  }

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_pairMultADAGradTile
(JNIEnv *env, jobject obj, jint nrows, jint bncols, jint brows1, jint brows2, jobject jA, jint lda, jint aroff, jint acoff,
 jobject jBdata, jobject jBir, jobject jBjc, jint broff, jint bcoff, jint transpose, 
 jobject jMM, jint ldmm, jobject jSumsq, jobject jMask, jint maskrows, jobject jlrate, jint lrlen,
 jobject jvexp, jint vexplen, jobject jtexp, jint texplen, jfloat istep, jint addgrad, jfloat epsilon)
  {
    float *A = (float*)getPointer(env, jA);
    float *Bdata = (float*)getPointer(env, jBdata);
    int *Bir = (int*)getPointer(env, jBir);
    int *Bjc = (int*)getPointer(env, jBjc);
    float *MM = (float*)getPointer(env, jMM);
    float *Sumsq = (float*)getPointer(env, jSumsq);
    float *Mask = (float*)getPointer(env, jMask);
    float *lrate = (float*)getPointer(env, jlrate);
    float *vexp = (float*)getPointer(env, jvexp);
    float *texp = (float*)getPointer(env, jtexp);

    return pairMultADAGradTile(nrows, bncols, brows1, brows2, A, lda, aroff, acoff, Bdata, Bir, Bjc, broff, bcoff, transpose, 
                               MM, ldmm, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
  }

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_CUMACH_linComb
(JNIEnv *env, jobject obj, jobject jX, jfloat wx, jobject jY, jfloat wy, jobject jZ, jint len)
  {
    float *X = (float*)getPointer(env, jX);
    float *Y = (float*)getPointer(env, jY);
    float *Z = (float*)getPointer(env, jZ);

    return linComb(X, wx, Y, wy, Z, len);
  }
}
