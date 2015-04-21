#include <jni.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecPos
(JNIEnv *env, jobject obj, jint nrows, jint ncols, const jint skip, jintArray jW, jintArray jLB, jintArray jUB,
 jfloatArray jA, jfloatArray jB, jfloat lrate, jint nthreads)
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
    int i, j, k, c, ia, ib, coff;
    float cv;
    float * Atmp = (float *)malloc(nrows*sizeof(float));  

    for (i = istart; i < iend; i++) {
      ia = nrows * W[i];
      if (ia >= 0) {
        for (c = 0; c < nrows; c++) {
          Atmp[c] = 0;
        }
        for (j = LB[i]; j <= UB[i]; j++) {
          cv = 0;
          if (j != 0 && i + j >= 0 && i + j < ncols) {
            ib = nrows * W[i + j];
            if (ib >= 0) {
              for (c = 0; c < nrows; c++) {
                cv += A[c + ia] * B[c + ib];
              }

              if (cv > 16.0f) {
                cv = 1.0f;
              } else if (cv < -16.0f) {
                cv = 0.0f;
              } else {
                cv = exp(cv);
                cv = 1.0f / (1.0f + cv);
              }

              cv = lrate * (1.0f - cv);
              for (c = 0; c < nrows; c++) {
                Atmp[c] += cv * B[c + ib];
                B[c + ib] += cv * A[c + ia];
              }
            }
          }
        }
        for (c = 0; c < nrows; c++) {
          A[c + ia] += Atmp[c];
        }
      }
    }
    free(Atmp);
  }

  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jUB, UB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jLB, LB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jW, W, 0);
}

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecNeg
(JNIEnv *env, jobject obj, jint nrows, jint ncols, const jint nwa, const jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jA, jfloatArray jB, jfloat lrate, jint nthreads)
{
  int ithread;
  int * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  int * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  float * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  float * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));

#pragma omp parallel for
  for (ithread = 0; ithread < nthreads; ithread++) {
    int i, j, k, c, ia, ib, ja;
    float cv;
    int istart = (1L * ithread * ncols)/nthreads;
    int iend = (1L * (ithread+1) * ncols)/nthreads;
    float * Atmp = (float *)malloc(nwa*nrows*sizeof(float));  
    float * Btmp = (float *)malloc(nrows*sizeof(float));  

    for (i = istart; i < iend; i++) {
      for (j = 0; j < nwa; j++) {
        ja = j * nrows;
        for (c = 0; c < nrows; c++) {
          Atmp[c + ja] = 0;
        }
      }
      for (k = 0; k < nwb; k++) {
        ib = nrows * WB[k+i*nwb];
        for (c = 0; c < nrows; c++) {
          Btmp[c] = 0;
        }
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
            cv = 1.0f / (1.0f + cv);
          } 

          cv = - lrate * cv;
          ja = j * nrows;
          for (c = 0; c < nrows; c++) {
            Btmp[c] += cv * A[c + ia];
            Atmp[c + ja] += cv * B[c + ib];
          }
        }
        for (c = 0; c < nrows; c++) {
          B[c + ib] += Btmp[c];
        }
      }
      for (j = 0; j < nwa; j++) {
        ja = j * nrows;
        ia = nrows * WA[j+i*nwa];
        for (c = 0; c < nrows; c++) {
          A[c + ia] += Atmp[c + ja];
        }
      }
    }
    free(Btmp);
    free(Atmp);
  }

  (*env)->ReleasePrimitiveArrayCritical(env, jB, B, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jA, A, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWB, WB, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, jWA, WA, 0);
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
