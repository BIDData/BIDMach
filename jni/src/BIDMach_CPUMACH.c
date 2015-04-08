#include <jni.h>
#include <mkl.h>
#include <omp.h>

JNIEXPORT void JNICALL Java_edu_berkeley_bid_CPUMACH_word2vecFwd
(JNIEnv *env, jobject obj, jint nrows, jint ncols, const jint nwa, const jint nwb, jintArray jWA, jintArray jWB, 
 jfloatArray jA, jfloatArray jB, jfloatArray jC)
{
  int i, j, k, c, ia, ib, coff;
  float sum;
  jint * WA = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWA, JNI_FALSE));
  jint * WB = (jint *)((*env)->GetPrimitiveArrayCritical(env, jWB, JNI_FALSE));
  jfloat * A = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jA, JNI_FALSE));
  jfloat * B = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jB, JNI_FALSE));
  jfloat * C = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, jC, JNI_FALSE));

#pragma omp parallel for
  for (i = 0; i < ncols; i++) {
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
