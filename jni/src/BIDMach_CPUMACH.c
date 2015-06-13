#include <jni.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>

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
