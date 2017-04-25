#include <jni.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#ifdef __GNUC__
#define __forceinline __attribute__((always_inline)) inline
#endif

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecPos
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint skip, jintArray jW, jintArray jLB, jintArray jUB,
 jfloatArray jA, jfloatArray jB, jfloat lrate, jfloat vexp, jint nthreads)
{
  int ithread;
  int * W = (jint *)((*env)->GetPrimitiveArrayCritical(env, jW, JNI_FALSE));
  int * LB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jLB, JNI_FALSE));
  int * UB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jUB, JNI_FALSE));
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;
    int i, j, k, c, ia, ib, coff, itmp;
    float cv, ascale, bscale;
    float * daa = (float *)malloc(nrows*sizeof(float));  

    for (i = istart; i < iend; i++) {
      itmp = W[i];
      ia = nrows * itmp;
      if (ia >= 0) {
        ascale = pow(1+itmp, vexp);
        for (c = 0; c < nrows; c++) {
          daa[c] = 0;
        }
        for (j = LB[i]; j <= UB[i]; j++) {
          if (j != 0 && i + j >= 0 && i + j < ncols) {
            itmp = W[i + j];
            ib = nrows * itmp;
            if (ib >= 0) {
              bscale = pow(1+itmp, vexp);
              cv = 0;
              for (c = 0; c < nrows; c++) {
                cv += A[c + ia] * B[c + ib];
              }

              if (cv > 16.0f) {
                cv = 1.0f;
              } else if (cv < -16.0f) {
                cv = 0.0f;
              } else {
                cv = exp(cv);
                cv = cv / (1.0f + cv);
              }

              cv = lrate * (1.0f - cv);
              for (c = 0; c < nrows; c++) {
                daa[c] += ascale * cv * B[c + ib];
                B[c + ib] += bscale * cv * A[c + ia];
              }
            }
          }
        }
        for (c = 0; c < nrows; c++) {
          A[c + ia] += daa[c];
        }
      }
    }
    free(daa);
  }

  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jUB, UB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jLB, LB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jW, W, 0);
}

void mapIndx(int *mm, int *ii, int *ismine, int *ishead, int indx, int islice, int nslices, int nhead, int maxcols, int nrows, int offset)
{
  int newi = indx;
  if (indx >= nhead) newi = ((indx - nhead) / nslices + nhead);                     // new column index
  *mm = newi / maxcols + offset;                                 // which matrix are we in? 
  *ismine = (indx >= nhead) && (indx % nslices == islice);
  *ishead = (indx < nhead);
  *ii = nrows * (newi - (*mm) * maxcols);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecPosSlice
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint skip, jintArray jW, jintArray jLB, jintArray jUB,
 jobjectArray jMM, jfloat lrate, jfloat vexp, jint nthreads,
 jint islice, jint nslices, jint maxCols, jint nHead, jint dualMode, jint doHead)
{
  int ix, ithread;
  int offset = 0;
  int * W = (jint *)((*env)->GetPrimitiveArrayCritical(env, jW, JNI_FALSE));
  int * LB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jLB, JNI_FALSE));
  int * UB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jUB, JNI_FALSE));
  int nelems = (*env)->GetArrayLength(env, jMM);
  jfloatArray *X = malloc(nelems * sizeof(jfloatArray));
  float **Y = malloc(nelems * sizeof(float *));
  float *A, *B;
  if (dualMode) offset = 1;
  for (ix = 0; ix < nelems; ix++) {
     X[ix] = (jfloatArray)((*env)->GetObjectArrayElement(env, jMM, ix));
     Y[ix] = (float *)((*env)->GetPrimitiveArrayCritical(env, X[ix], JNI_FALSE));
  }

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;
    int i, j, k, c, ia, ib, iac, ibc, coff;
    int aismine, bismine, aishead, bishead, ma, mb;
    float cv, ascale, bscale;
    int touched; 
    float * daa = (float *)malloc(nrows*sizeof(float));  

    for (i = istart; i < iend; i++) {
      iac = W[i];
      if (iac >= 0) {
        mapIndx(&ma, &ia, &aismine, &aishead, iac, islice, nslices, nHead, maxCols, nrows, offset);
        A = Y[2*ma+1];
        ascale = pow(1+iac, vexp);
        for (c = 0; c < nrows; c++) {
          daa[c] = 0;
        }
        touched = 0;
        for (j = LB[i]; j <= UB[i]; j++) {
          if (j != 0 && i + j >= 0 && i + j < ncols) {
            ibc = W[i + j];
            if (ibc >= 0) {
              mapIndx(&mb, &ib, &bismine, &bishead, ibc, islice, nslices, nHead, maxCols, nrows, offset);
              B = Y[2*mb];
              bscale = pow(1+ibc, vexp);
              if ((doHead > 1 && aishead && bishead) || (aismine && bishead) || (bismine && aishead) || (aismine && bismine)) {
                touched = 1;
                cv = 0;
                for (c = 0; c < nrows; c++) {
                  cv += A[c + ia] * B[c + ib];
                }

                if (cv > 16.0f) {
                  cv = 1.0f;
                } else if (cv < -16.0f) {
                  cv = 0.0f;
                } else {
                  cv = exp(cv);
                  cv = cv / (1.0f + cv);
                }

                cv = lrate * (1.0f - cv);
                for (c = 0; c < nrows; c++) {
                  daa[c] += ascale * cv * B[c + ib];
                }
                if (bismine || (bishead && doHead)) {
                  for (c = 0; c < nrows; c++) {
                    B[c + ib] += bscale * cv * A[c + ia];
                  }
                }
              }
            }
          }
        }
        if (touched && (aismine || (aishead && doHead))) {
          for (c = 0; c < nrows; c++) {
            A[c + ia] += daa[c];
          }
        }
      }
    }
    free(daa);
  }

  for (ix = nelems-1; ix >= 0; ix--) {
    (*env)->ReleasePrimitiveArrayCritical(env, X[ix], Y[ix], 0);
  }
  (*env)->ReleasePrimitiveArrayCritical(env, jUB, UB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jLB, LB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jW, W, 0);
  free(Y);
  free(X);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecNeg
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jA, jfloatArray jB, jfloat lrate, jfloat vexp, jint nthreads)
{
  int ithread;
  int * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  int * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int i, j, k, c, ia, ib, ja, itmp;
    float cv, ascale, bscale;
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;
    float * daa = (float *)malloc(nwa*nrows*sizeof(float));  
    float * dbb = (float *)malloc(nrows*sizeof(float));  

    for (i = istart; i < iend; i++) {
      for (j = 0; j < nwa; j++) {
        ja = j * nrows;
        for (c = 0; c < nrows; c++) {
          daa[c + ja] = 0;
        }
      }
      for (k = 0; k < nwb; k++) {
        itmp = WB[k+i*nwb];
        ib = nrows * itmp;
        bscale = pow(1+itmp, vexp);
        for (c = 0; c < nrows; c++) {
          dbb[c] = 0;
        }
        for (j = 0; j < nwa; j++) {
          itmp = WA[j+i*nwa];
          ia = nrows * itmp;
          ascale = pow(1+itmp, vexp);
          cv = 0;
          for (c = 0; c < nrows; c++) {
            cv += A[c + ia] * B[c + ib];
          }

          if (cv > 16.0f) {
            cv = 1.0f;
          } else if (cv < -16.0f) {
            cv = 0.0f;
          } else {
            cv = exp(cv);
            cv = cv / (1.0f + cv);
          } 

          cv = - lrate * cv;
          ja = j * nrows;
          for (c = 0; c < nrows; c++) {
            dbb[c] += bscale * cv * A[c + ia];
            daa[c + ja] += ascale * cv * B[c + ib];
          }
        }
        for (c = 0; c < nrows; c++) {
          B[c + ib] += dbb[c];
        }
      }
      for (j = 0; j < nwa; j++) {
        ja = j * nrows;
        ia = nrows * WA[j+i*nwa];
        for (c = 0; c < nrows; c++) {
          A[c + ia] += daa[c + ja];
        }
      }
    }
    free(dbb);
    free(daa);
  }

  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWB, WB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWA, WA, 0);
}


JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecNegSlice
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jMM, jfloat lrate, jfloat vexp, jint nthreads,
 jint islice, jint nslices, jint maxCols, jint nHead, jint dualMode, jint doHead)
{
  int ix, ithread, offset = 0;
  int * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  int * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  float *A, *B;
  int nelems = (*env)->GetArrayLength(env, jMM);
  jfloatArray *X = malloc(nelems * sizeof(jfloatArray));
  float **Y = malloc(nelems * sizeof(float *));
  if (dualMode) offset = 1;
  for (ix = 0; ix < nelems; ix++) {
     X[ix] = (jfloatArray)((*env)->GetObjectArrayElement(env, jMM, ix));
     Y[ix] = (float *)((*env)->GetPrimitiveArrayCritical(env, X[ix], JNI_FALSE));
  }

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int i, j, k, c, ia, ib, iac, ibc, ja;
    float cv, ascale, bscale;
    int aismine, bismine, aishead, bishead, ma, mb;
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;
    float * daa = (float *)malloc(nwa*nrows*sizeof(float));  
    float * dbb = (float *)malloc(nrows*sizeof(float));  

    for (i = istart; i < iend; i++) {
      for (j = 0; j < nwa; j++) {
        ja = j * nrows;
        for (c = 0; c < nrows; c++) {
          daa[c + ja] = 0;
        }
      }
      for (k = 0; k < nwb; k++) {
        ibc = WB[k+i*nwb];
        mapIndx(&mb, &ib, &bismine, &bishead, ibc, islice, nslices, nHead, maxCols, nrows, offset);
        B = Y[2*mb];
        bscale = pow(1+ibc, vexp);
        for (c = 0; c < nrows; c++) {
          dbb[c] = 0;
        }
        for (j = 0; j < nwa; j++) {
          iac = WA[j+i*nwa];
          mapIndx(&ma, &ia, &aismine, &aishead, iac, islice, nslices, nHead, maxCols, nrows, offset);
          A = Y[2*ma+1];
          ascale = pow(1+iac, vexp);
          if ((doHead > 1 && aishead && bishead) || (aismine && bishead) || (bismine && aishead) || (aismine && bismine)) {
            cv = 0;
            for (c = 0; c < nrows; c++) {
              cv += A[c + ia] * B[c + ib];
            }

            if (cv > 16.0f) {
              cv = 1.0f;
            } else if (cv < -16.0f) {
              cv = 0.0f;
            } else {
              cv = exp(cv);
              cv = cv / (1.0f + cv);
            } 

            cv = - lrate * cv;
            ja = j * nrows;
            for (c = 0; c < nrows; c++) {
              dbb[c] += bscale * cv * A[c + ia];
              daa[c + ja] += ascale * cv * B[c + ib];
            }
          }
        }
        if (bismine || (bishead && doHead)) {
          for (c = 0; c < nrows; c++) {
            B[c + ib] += dbb[c];
          }
        }
      }
      for (j = 0; j < nwa; j++) {
        ja = j * nrows;
        iac = WA[j+i*nwa];
        mapIndx(&ma, &ia, &aismine, &aishead, iac, islice, nslices, nHead, maxCols, nrows, offset);
        A = Y[2*ma+1];
        if (aismine || (aishead && doHead)) {
          for (c = 0; c < nrows; c++) {
            A[c + ia] += daa[c + ja];
          }
        }
      }
    }
    free(dbb);
    free(daa);
  }
  for (ix = nelems-1; ix >= 0; ix--) {
    (*env)->ReleasePrimitiveArrayCritical(env, X[ix], Y[ix], 0);
  }
  (*env)->ReleasePrimitiveArrayCritical(env, jWB, WB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWA, WA, 0);
  free(Y);
  free(X);
}

JNIEXPORT jdouble JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecEvalPos
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint skip, jintArray jW, jintArray jLB, jintArray jUB,
 jfloatArray jA, jfloatArray jB, jint nthreads)
{
  int i, ithread;
  int * W = (jint *)((*env)->GetPrimitiveArrayCritical(env, jW, JNI_FALSE));
  int * LB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jLB, JNI_FALSE));
  int * UB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jUB, JNI_FALSE));
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));
  double * pv = (double *)malloc(nthreads * sizeof(double));
  double sum;

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;
    int i, j, k, c, ia, ib, coff;
    float cv;
    double dv = 0;
 
    for (i = istart; i < iend; i++) {
      ia = nrows * W[i];
      if (ia >= 0) {
        for (j = LB[i]; j <= UB[i]; j++) {
          if (j != 0 && i + j >= 0 && i + j < ncols) {
            ib = nrows * W[i + j];
            if (ib >= 0) {
              cv = 0;
              for (c = 0; c < nrows; c++) {
                cv += A[c + ia] * B[c + ib];
              }
              if (cv > 16.0f) {
                cv = 1.0f;
              } else if (cv < -16.0f) {
                cv = 0.0f;
              } else {
                cv = exp(cv);
                cv = cv / (1.0f + cv);
              }

              dv += log(fmax((double)cv, 1.0e-40));
            }
          }
        }
      }
    }
    pv[ithread] = dv;
  }
  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jUB, UB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jLB, LB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jW, W, 0);
  for (i = 0; i < nthreads; i++) {
    sum += pv[i];
  }
  free(pv);
  return sum;
}

JNIEXPORT jdouble JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecEvalNeg
(JNIEnv *env, jobject obj, jint nrows, jint ncols, const jint nwa, const jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jA, jfloatArray jB, jint nthreads)
{
  int i, ithread;
  int * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  int * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));
  double * pv = (double *)malloc(nthreads * sizeof(double));
  double sum;

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int i, j, k, c, ia, ib, ja;
    float cv;
    double dv = 0;
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;

    for (i = istart; i < iend; i++) {
      for (k = 0; k < nwb; k++) {
        ib = nrows * WB[k+i*nwb];
        for (j = 0; j < nwa; j++) {
          ia = nrows * WA[j+i*nwa];
          cv = 0;
          for (c = 0; c < nrows; c++) {
            cv += A[c + ia] * B[c + ib];
          }

          if (cv > 16.0f) {
            cv = 1.0f;
          } else if (cv < -16.0f) {
            cv = 0.0f;
          } else {
            cv = exp(cv);
            cv = cv / (1.0f + cv);
          } 
          dv += log(fmax(1.0 - (double)cv, 1.0e-40));
        }
      }
    }
    pv[ithread] = dv;
  }

  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWB, WB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWA, WA, 0);
  for (i = 0; i < nthreads; i++) {
    sum += pv[i];
  }
  free(pv);
  return sum;
}


JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecFwd
(JNIEnv *env, jobject obj, jint nrows, jint ncols, const jint nwa, const jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jA, jfloatArray jB, jfloatArray jC)
{
  int i;
  jint * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  jint * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  jfloat * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  jfloat * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));
  jfloat * C = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jC, JNI_FALSE));

#pragma omp parallel for
  for (i = 0; i < ncols; i++) {
    int j, k, c, ia, ib, coff;
    float sum;
    for (j = 0; j < nwa; j++) {
      ia = nrows*WA[j+i*nwa];
      for (k = 0; k < nwb; k++) {
        ib = nrows*WB[k+i*nwb];
        sum = 0;
        for (c = 0; c < nrows; c++) {
          sum += A[c + ia] * B[c + ib];
        }
        coff = nwa * (k + nwb * i);
        C[j + coff] = sum;
      }
    } 
  }

  (*env)->ReleasePrimitiveArrayCritical(env, jC, C, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWB, WB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWA, WA, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecBwd
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nwa, jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jA, jfloatArray jB, jfloatArray jC, jfloat lrate)
{
  int i, j, k, c;
  float cv;
  int ia, ib;
  jint * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  jint * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  jfloat * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  jfloat * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));
  jfloat * C = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jC, JNI_FALSE));

#pragma omp parallel for
  for (i = 0; i < ncols; i++) {
    for (j = 0; j < nwa; j++) {
      ia = nrows*WA[j+i*nwa];
      for (c = 0; c < nrows; c++) {
        A[c + ia] = 0;
      }
      for (k = 0; k < nwb; k++) {
        ib = nrows*WB[k+i*nwb];
        cv = lrate * C[j + nwa * (k + nwb * i)];
        for (c = 0; c < nrows; c++) {
          A[c + ia] += cv * B[c + ib];
        }
      }
    }
    for (k = 0; k < nwb; k++) {
      ib = nrows*WB[k+i*nwb];
      for (c = 0; c < nrows; c++) {
        B[c + ib] = 0;
      }
      for (j = 0; j < nwa; j++) {
        ia = nrows*WA[j+i*nwa];
        cv = lrate * C[j + nwa * (k + nwb * i)];
        for (c = 0; c < nrows; c++) {
          B[c + ib] += cv * A[c + ia];
        }
      }
    }
  }
  (*env)->ReleasePrimitiveArrayCritical(env, jC, C, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWB, WB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWA, WA, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_testarrays
(JNIEnv *env, jobject obj, jobjectArray arr) 
{
  int i;
  int nelems = (*env)->GetArrayLength(env, arr);
  jfloatArray *X = malloc(nelems * sizeof(jfloatArray));
  jfloat **Y = malloc(nelems * sizeof(jfloat *));
  for (i = 0; i < nelems; i++) {
     X[i] = (jfloatArray)((*env)->GetObjectArrayElement(env, arr, i));
     Y[i] = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, X[i], JNI_FALSE));
  }
  printf("n=%d, v=%f, u=%f\n", nelems, Y[0][0], Y[1][0]);
  fflush(stdout);
  for (i = 0; i < nelems; i++) {
    (*env)->ReleasePrimitiveArrayCritical(env, X[i], Y[i], 0);
  }
  free(X);
  free(Y);
}

#define APPLYCFN(fn)                                        \
  for (i = istart; i < iend; i++) {                         \
    float x = A[i];                                         \
    B[i] = (fn);         				    \
  }


#define APPLYCOP(fn)                                        \
  for (i = istart; i < iend; i++) {                         \
    float x = A[i];                                         \
    float y = B[i];                                         \
    C[i] = (fn);         				    \
  }

#define SIGMOIDN 0
#define TANHN 1
#define SOFTPLUSN 2

#define SIGMOIDX (x > 20.0f) ? 1.0f : ((x < -80.0f) ? 0.0f : 1.0f/(1.0f + exp(-x)))

#define SOFTPLUSX (x > 20.0f) ? x : ((x < -20.0f) ? 0.0f : log(1.0f + exp(x)))

#define SIGMOIDY (y * (x - x * x))

#define TANHY (y * (1.0f - x * x))

#define SOFTPLUSY y * ((x > 20.0f) ? 1.0f : ((x < -80.0f) ? 0.0f : 1.0f/(1.0f + exp(-x))))
  

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_applyfwd
(JNIEnv *env, jobject obj, jfloatArray jA, jfloatArray jB, jint ifn, jint n, jint nthreads)
{
  int ithread;
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int istart = (1L * ithread * n)/nthreads;
    int iend = (1L * (ithread+1) * n)/nthreads;
    int i;
    switch (ifn) {
    case SIGMOIDN: APPLYCFN(SIGMOIDX); break;
    case SOFTPLUSN: APPLYCFN(SOFTPLUSX); break;
    }
  }
  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_applyderiv
(JNIEnv *env, jobject obj, jfloatArray jA, jfloatArray jB, jfloatArray jC, jint ifn, jint n, jint nthreads)
{
  int ithread;
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));
  float * C = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jC, JNI_FALSE));

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int istart = (1L * ithread * n)/nthreads;
    int iend = (1L * (ithread+1) * n)/nthreads;
    int i;
    switch (ifn) {
    case SIGMOIDN: APPLYCOP(SIGMOIDY); break;
    case TANHN: APPLYCOP(TANHY); break;
    case SOFTPLUSN: APPLYCOP(SOFTPLUSY); break;
    }
  }
  (*env)->ReleasePrimitiveArrayCritical(env, jC, C, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_multADAGrad
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint nnz, jfloatArray jA, jfloatArray jBdata, jintArray jBir, jintArray jBjc, 
 jfloatArray jMM, jfloatArray jSumsq, jfloatArray jMask, int maskrows, jfloatArray jlrate, jint lrlen,
 jfloatArray jvexp, jint vexplen, jfloatArray jtexp, jint texplen, jfloat istep, jint addgrad, jfloat epsilon, jint biasv, jint nbr)
{
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * Bdata = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jBdata, JNI_FALSE));
  int * Bir = (jint *)((*env)->GetPrimitiveArrayCritical(env, jBir, JNI_FALSE));
  int * Bjc = (jint *)((*env)->GetPrimitiveArrayCritical(env, jBjc, JNI_FALSE));
  float * MM = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jMM, JNI_FALSE));
  float * Sumsq = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jSumsq, JNI_FALSE));
  float * lrate = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jlrate, JNI_FALSE));
  float * vexp = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jvexp, JNI_FALSE));
  float * texp = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jtexp, JNI_FALSE));
  float * Mask = NULL;
  int i;
  int ioff = Bjc[0];
  if (jMask != NULL) Mask = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jMask, JNI_FALSE));

#pragma omp parallel for
  for (i = 0; i < ncols; i++) {
    int jstart = Bjc[i] - ioff;
    int jend = Bjc[i+1] - ioff;
    int j;
    if (Mask != NULL || lrlen > 1 || vexplen > 1 || texplen > 1) {
      for (j = jstart; j < jend ; j++) {
        float bval = Bdata[j];
        int ival = Bir[j] - ioff;
        int k;
        for (k = 0; k < nrows; k++) {
          float lr, ve, te, pve, ste, ngrad;
          float grad = A[k+i*nrows]*bval;
          int ihere = k+ival*nrows;
          Sumsq[ihere] += grad*grad + epsilon;
          if (addgrad) {
            lr =  (lrlen > 1) ? lrate[k] : lrate[0];
            ve =  (vexplen > 1) ? vexp[k] : vexp[0];
            te =  (texplen > 1) ? texp[k] : texp[0];
            pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
            ste = pow(istep, te);
            ngrad = grad * lr * ste / pve;
            MM[ihere] += ngrad;
          }
          if (Mask != NULL) {
            if (maskrows > 1) {
              if (Mask[ihere] == 0) MM[ihere] = 0;
            } else {
              if (Mask[ival] == 0) MM[ihere] = 0;
            }
          }
        }
      }
      if (biasv > 0) {
        int ival = nbr;
        int k;
        for (k = 0; k < nrows; k++) {
          float lr, ve, te, pve, ste, ngrad;
          float grad = A[k+i*nrows];
          int ihere = k+ival*nrows;
          Sumsq[ihere] += grad*grad + epsilon;
          if (addgrad) {
            lr =  (lrlen > 1) ? lrate[k] : lrate[0];
            ve =  (vexplen > 1) ? vexp[k] : vexp[0];
            te =  (texplen > 1) ? texp[k] : texp[0];
            pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
            ste = pow(istep, te);
            ngrad = grad * lr * ste / pve;
            MM[ihere] += ngrad;
          }
          if (Mask != NULL) {
            if (maskrows > 1) {
              if (Mask[ihere] == 0) MM[ihere] = 0;
            } else {
              if (Mask[ival] == 0) MM[ihere] = 0;
            }
          }
        }
      }
    } else {
      float lr, ve, te, pve, ste, ngrad;
      lr =  lrate[0];
      ve =  vexp[0];
      te =  texp[0];
      for (j = jstart; j < jend ; j++) {
        float bval = Bdata[j];
        int ival = Bir[j] - ioff;
        int k;
        if (addgrad && ve == 0.5f && te == 0.5f) {
          for (k = 0; k < nrows; k++) {
            float grad = A[k+i*nrows]*bval;
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            pve = sqrt(Sumsq[ihere]);
            ngrad = grad * lr / pve;
            MM[ihere] += ngrad;
          }
        } else {
          for (k = 0; k < nrows; k++) {
            float grad = A[k+i*nrows]*bval;
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            if (addgrad) {
              pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
              ste = pow(istep, te);
              ngrad = grad * lr * ste / pve;
              MM[ihere] += ngrad;
            }
          }
        }
      }
      if (biasv > 0) {
        int ival = nbr;
        int k;
        if (addgrad && ve == 0.5f && te == 0.5f) {
          for (k = 0; k < nrows; k++) {
            float grad = A[k+i*nrows];
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            pve = sqrt(Sumsq[ihere]);
            ngrad = grad * lr / pve;
            MM[ihere] += ngrad;
          }
        } else {
          for (k = 0; k < nrows; k++) {
            float grad = A[k+i*nrows];
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            if (addgrad) {
              pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
              ste = pow(istep, te);
              ngrad = grad * lr * ste / pve;
              MM[ihere] += ngrad;
            }
          }
        }
      }
    }
  }
  if (Mask != NULL) (*env)->ReleasePrimitiveArrayCritical(env, jMask, Mask, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jtexp, texp, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jvexp, vexp, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jlrate, lrate, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jSumsq, Sumsq, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jMM, MM, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBjc, Bjc, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBir, Bir, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBdata, Bdata, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_multADAGradTile
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint y, jint x, jint nnz, jfloatArray jA, jint lda, jfloatArray jBdata, jintArray jBir, jintArray jBjc, 
 jfloatArray jMM, jfloatArray jSumsq, jfloatArray jMask, int maskrows, jfloatArray jlrate, jint lrlen,
 jfloatArray jvexp, jint vexplen, jfloatArray jtexp, jint texplen, jfloat istep, jint addgrad, jfloat epsilon, jint biasv, jint nbr)
{
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * Bdata = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jBdata, JNI_FALSE));
  int * Bir = (jint *)((*env)->GetPrimitiveArrayCritical(env, jBir, JNI_FALSE));
  int * Bjc = (jint *)((*env)->GetPrimitiveArrayCritical(env, jBjc, JNI_FALSE));
  float * MM = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jMM, JNI_FALSE));
  float * Sumsq = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jSumsq, JNI_FALSE));
  float * lrate = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jlrate, JNI_FALSE));
  float * vexp = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jvexp, JNI_FALSE));
  float * texp = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jtexp, JNI_FALSE));
  float * Mask = NULL;
  int i;
  int ioff = Bjc[0];
  if (jMask != NULL) Mask = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jMask, JNI_FALSE));

#pragma omp parallel for
  for (i = 0; i < ncols; i++) {
    int jstart = Bjc[i] - ioff;
    int jend = Bjc[i+1] - ioff;
    int j;
    if (Mask != NULL || lrlen > 1 || vexplen > 1 || texplen > 1) {
      for (j = jstart; j < jend ; j++) {
        float bval = Bdata[j];
        int ival = Bir[j] - ioff - x;
        if (ival >= 0 && ival < ncols) {
          int k;
          for (k = 0; k < nrows; k++) {
            float lr, ve, te, pve, ste, ngrad;
            float grad = A[y+k+i*lda]*bval;
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            if (addgrad) {
              lr =  (lrlen > 1) ? lrate[k+y] : lrate[0];
              ve =  (vexplen > 1) ? vexp[k+y] : vexp[0];
              te =  (texplen > 1) ? texp[k+y] : texp[0];
              pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
              ste = pow(istep, te);
              ngrad = grad * lr * ste / pve;
              MM[ihere] += ngrad;
            }
            if (Mask != NULL) {
              if (maskrows > 1) {
                if (Mask[ihere] == 0) MM[ihere] = 0;
              } else {
                if (Mask[ival] == 0) MM[ihere] = 0;
              }
            }
          }
        }
      }
      if (biasv > 0) {
        int ival = nbr;
        int k;
        for (k = 0; k < nrows; k++) {
          float lr, ve, te, pve, ste, ngrad;
          float grad = A[k+y+i*lda];
          int ihere = k+ival*nrows;
          Sumsq[ihere] += grad*grad + epsilon;
          if (addgrad) {
            lr =  (lrlen > 1) ? lrate[k+y] : lrate[0];
            ve =  (vexplen > 1) ? vexp[k+y] : vexp[0];
            te =  (texplen > 1) ? texp[k+y] : texp[0];
            pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
            ste = pow(istep, te);
            ngrad = grad * lr * ste / pve;
            MM[ihere] += ngrad;
          }
          if (Mask != NULL) {
            if (maskrows > 1) {
              if (Mask[ihere] == 0) MM[ihere] = 0;
            } else {
              if (Mask[ival] == 0) MM[ihere] = 0;
            }
          }
        }
      }
    } else {
      float lr, ve, te, pve, ste, ngrad;
      lr =  lrate[0];
      ve =  vexp[0];
      te =  texp[0];
      for (j = jstart; j < jend ; j++) {
        float bval = Bdata[j];
        int ival = Bir[j] - ioff - x;
        if (ival >= 0 && ival < ncols) {
          int k;
          if (addgrad && ve == 0.5f && te == 0.5f) {
            for (k = 0; k < nrows; k++) {
              float grad = A[k+y+i*lda]*bval;
              int ihere = k+ival*nrows;
              Sumsq[ihere] += grad*grad + epsilon;
              pve = sqrt(Sumsq[ihere]);
              ngrad = grad * lr / pve;
              MM[ihere] += ngrad;
            }
          } else {
            for (k = 0; k < nrows; k++) {
              float grad = A[k+y+i*nrows]*bval;
              int ihere = k+ival*nrows;
              Sumsq[ihere] += grad*grad + epsilon;
              if (addgrad) {
                pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
                ste = pow(istep, te);
                ngrad = grad * lr * ste / pve;
                MM[ihere] += ngrad;
              }
            }
          }
        }
      }
      if (biasv > 0) {
        int ival = nbr;
        int k;
        if (addgrad && ve == 0.5f && te == 0.5f) {
          for (k = 0; k < nrows; k++) {
            float grad = A[k+y+i*lda];
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            pve = sqrt(Sumsq[ihere]);
            ngrad = grad * lr / pve;
            MM[ihere] += ngrad;
          }
        } else {
          for (k = 0; k < nrows; k++) {
            float grad = A[k+y+i*nrows];
            int ihere = k+ival*nrows;
            Sumsq[ihere] += grad*grad + epsilon;
            if (addgrad) {
              pve = (ve == 0) ? 1.0f : pow(Sumsq[ihere] * istep, ve);
              ste = pow(istep, te);
              ngrad = grad * lr * ste / pve;
              MM[ihere] += ngrad;
            }
          }
        }
      }
    }
  }
  if (Mask != NULL) (*env)->ReleasePrimitiveArrayCritical(env, jMask, Mask, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jtexp, texp, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jvexp, vexp, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jlrate, lrate, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jSumsq, Sumsq, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jMM, MM, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBjc, Bjc, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBir, Bir, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBdata, Bdata, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
}

__forceinline long long __pairembed(long long r1x, int r2x) {
  long long r1 = r1x+1;
  int r2 = r2x+1;
  float loc1 = (float) r1;
  float loc2 = (float) r2;
  int nbits1 = ((*(int *)(&loc1)) >> 23) - 126;
  int nbits2 = ((*(int *)(&loc2)) >> 23) - 126;
  int len = nbits1 + nbits2 - 2;
  float loc3 = (float) len; 
  int lenbits = 0;
  long long x;
  if (len > 1) lenbits = ((*(int *)(&loc3)) >> 23) - 127;
  r2 = r2 & ((1 << (nbits2-1)) - 1);
  x = (((r1 << (nbits2-1)) | r2) << lenbits) | (nbits2-1);
  return (x-2) >= 0 ? (x-2) : 0;
}


__forceinline void __gupdate(float grad, int i, int ihere, int jhere, float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                      float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  float lr, ve, te, pve, ste, ngrad, ssq, ssqnew;
  ssq = Sumsq[ihere];
  ssqnew = hypotf(grad,ssq);
  Sumsq[ihere] += ssqnew - ssq;
  ssq = ssqnew * sqrtf(istep);

  if (addgrad) {
    lr =  (lrlen > 1) ? lrate[i] : lrate[0];
    ve =  (vexplen > 1) ? vexp[i] : vexp[0];
    te =  (texplen > 1) ? texp[i] : texp[0];
    pve = (ve == 0.5f) ? ssq : ((ve == 0) ? 1.0f : pow(ssq, 2*ve));
    ste = pow(istep, te);
    ngrad = grad * lr * ste / pve;
    MM[ihere] += ngrad;
  }
  if (Mask != NULL) {
    if (maskrows > 1) {
      if (Mask[ihere] == 0) MM[ihere] = 0;
    } else {
      if (Mask[jhere] == 0) MM[ihere] = 0;
    }
  }
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_pairMultADAGradTile
(JNIEnv *env, jobject obj, jint nrows, jint ncols, jint bound1, jint bound2, jfloatArray jA, jint lda, jint aroff, jint acoff, 
 jfloatArray jBdata, jintArray jBir, jintArray jBjc, jint broff, jint bcoff,
 jfloatArray jMM, jint ldmm, jfloatArray jSumsq, jfloatArray jMask, int maskrows, jfloatArray jlrate, jint lrlen,
 jfloatArray jvexp, jint vexplen, jfloatArray jtexp, jint texplen, jfloat istep, jint addgrad, jfloat epsilon, jint biasv, jint nbr)
{
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * Bdata = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jBdata, JNI_FALSE));
  int * Bir = (jint *)((*env)->GetPrimitiveArrayCritical(env, jBir, JNI_FALSE));
  int * Bjc = (jint *)((*env)->GetPrimitiveArrayCritical(env, jBjc, JNI_FALSE));
  float * MM = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jMM, JNI_FALSE));
  float * Sumsq = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jSumsq, JNI_FALSE));
  float * lrate = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jlrate, JNI_FALSE));
  float * vexp = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jvexp, JNI_FALSE));
  float * texp = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jtexp, JNI_FALSE));
  float * Mask = NULL;
  int i;
  int ioff = Bjc[0];
  if (jMask != NULL) Mask = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jMask, JNI_FALSE));

#pragma omp parallel for
  for (i = 0; i < ncols; i++) {
    int jstart = Bjc[i+bcoff] - ioff;
    int jend = Bjc[i+1+bcoff] - ioff;
    int j1, j2, k, ihere, jhere, ithere, jthere, doit, r1, r2;
    float grad, f1, f2, prod;
    long long rank;
    for (j1 = jstart; j1 < jend ; j1++) {
      f1 = Bdata[jstart + j1];                          // Get the feature
      r1 = Bir[jstart + j1]-broff-ioff;                 // And its row index
      rank = r1;
      if (r1 >= 0 && r1 < bound1) {
        for (k = 0; k < nrows; k++) {
          ihere = k + aroff + lda * (i + acoff);
          jhere = k + aroff;
          ithere = k + 2 * ldmm * rank;
          jthere = 2 * rank;
          grad = A[ihere] * f1;                        // raw gradient
          __gupdate(grad, jhere, ithere, jthere, MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
        }
        for (j2 = jstart; j2 < j1 ; j2++) {
          f2 = Bdata[jstart + j2];                     // Get the other feature
          r2 = Bir[jstart + j2]-broff-ioff;
          if (r2 >= 0) {
            rank = __pairembed(r1, r2);
            if (rank < bound2) {
              prod = f1 * f2;
              for (k = 0; k < nrows; k++) {
                ihere = k + aroff + lda * (i + acoff);
                jhere = k + aroff;
                ithere = ldmm + k + 2 * ldmm * rank;
                jthere = 1 + 2 * rank;
                grad = A[ihere] * prod;    // raw gradient
                __gupdate(grad, jhere, ithere, jthere, MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
              }
            }
          }
	}
      }
    }
  }
  if (Mask != NULL) (*env)->ReleasePrimitiveArrayCritical(env, jMask, Mask, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jtexp, texp, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jvexp, vexp, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jlrate, lrate, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jSumsq, Sumsq, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jMM, MM, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBjc, Bjc, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBir, Bir, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jBdata, Bdata, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
}
