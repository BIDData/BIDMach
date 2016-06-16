#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>
#include <MurmurHash.hpp>

#if __CUDA_ARCH__ >= 300
#define MAXXGRID 2147483647
#else
#define MAXXGRID 65535
#endif

void setsizes(int N, dim3 *gridp, int *nthreadsp);

__forceinline__ __device__ int solve1(int j) {
  float v = sqrtf((float)j);
#pragma unroll
  for (int k = 0; k < 5; k++) {
    v = v - (v*(v+1)-2*j)/(2*v+1);   // Newton iterations to find first index. 
  }
  return (int)(v+2e-5f);   
}

__forceinline__ __device__ void solvex(int n, int v, int &i, int &j) {
  int n1 = ((n >> 1) << 1) + 1;
  int n2 = (n + 1) >> 1;
  int even = (n1 != n);
  j = v / n1;
  i = v - n1 * j;
  if (j > i - even) {
    i = n1 - i - 1;
    j = n2 + n2 - j + 1;
  } else {
    i = i - even;
  }
}

// Feature hashing multiply and multiply-transpose.
// This one enumerates, hashes and multiplies all pairs of features.
//
// NOTE: The single-matrix version (hashmult) uses a fast lookup recurrence which is only valid up to 3000 base features per column (approx 4.5 million pairs)

// Given dense A and sparse B, for each column of B, enumerate all pairs of features, hash to a single feature index, and multiply by A into C

__global__ void __hashmult(int nrows, int nfeats, int bncols, int brows1, int brows2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose) {
  bool doit = false;
  int istart = ((long long)blockIdx.x) * bncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * bncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i];                                     // Range of nz rows in this column
    int jend = Bjc[i+1];
    int nr = jend - jstart;                                  // Number of nz rows
    int todo = nr * (nr + 1) / 2;                            // Number of pairs to process (including k,k pairs)
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = solve1(j);                                    // Compute the first and second indices
      int j2 = j - j1*(j1+1)/2; 
      //      int j1, j2;
      //      solvex(todo, j, j1, j2);
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1];                             // And their row indices
      int r2 = Bir[jstart + j2];
      long long rank = r1 + 1;
      float prod = f1;
      if (j1 == j2) {
        doit = (rank < brows1);
      } else {
        prod *= f2;
        rank *= r2 + 1;
        doit = (rank < brows2);
      }
      if (doit) {
        int ind = mmhash2(r1, r2, nfeats);                     // Hash the indices
        if (transpose > 0) {
          float sum = A[threadIdx.x + nrows * i] * prod;    // Do the product
          atomicAdd(&C[threadIdx.x + nrows * ind], sum);
        } else {
          float sum = A[threadIdx.x + nrows * ind] * prod;  // Do the product
          atomicAdd(&C[threadIdx.x + nrows * i], sum);
        }
      }
    }
  }
}

__forceinline__ __device__ int hash2(int a, int b, int modulus) {
  return  (((a * 453423453) + b) * 34143242142) % modulus;
}

#if __CUDA_ARCH__ >= 300

// This version is designed for few (or one) row in A. It allocates one warp per column

__global__ void __hashmult2(int nrows, int nfeats, int ncols, int brows1, int brows2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose) {
  bool doit = false;
  int istart = ((long long)blockIdx.x) * ncols / gridDim.x;
  int iend = ((long long)(blockIdx.x+1)) * ncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i];                                     // Range of nz rows in this column
    int jend = Bjc[i+1];
    int nr = jend - jstart;                                  // Number of nz rows
    for (int j1 = 0; j1 < nr; j1 += blockDim.x) {               // work on a block of data
      float f1 = 0;
      int r1 = -1;
      if (j1 + threadIdx.x < nr) {
        f1 = Bdata[jstart + j1 + threadIdx.x];                // Get the two features
        r1 = Bir[jstart + j1 + threadIdx.x];                  // And their row indices
      }
      for (int j2 = j1; j2 < nr; j2 += blockDim.x) {             // work on a block of data
        float f2 = 0;
        int r2 = -1;
        if (j2 + threadIdx.x < nr) {
          f2 = Bdata[jstart + j2 + threadIdx.x];
          r2 = Bir[jstart + j2 + threadIdx.x];
        }
        for (int k = 0; k < 32; k++) {
          float f2shift = __shfl(f2, k);
          int r2shift = __shfl(r2, k);
          if (j2 + k < nr && r1 >= 0) {
            long long rank = r1 + 1;
            float prod = f1;
            doit = false;
            if (j1 + threadIdx.x == j2 + k) {
              doit = (rank < brows1);
            } else if (j1 + threadIdx.x < j2 + k) {
              prod *= f2shift;
              rank *= r2shift + 1;
              doit = (rank < brows2);
            }
            if (doit) {
              int ind = mmhash2(r1, r2shift, nfeats);           // Hash the indices
              if (transpose > 0) {
                for (int m = 0; m < nrows; m++) {
                  float sum = A[m + nrows * i] * prod;    // Do the product
                  atomicAdd(&C[m + nrows * ind], sum);
                  //		  atomicAdd(&C[0], sum);
                }
              } else {
                for (int m = 0; m < nrows; m++) {
                  float sum = A[m + nrows * ind] * prod;  // Do the product
                  atomicAdd(&C[m + nrows * i], sum);
                  //		  atomicAdd(&C[0], sum);
                }
              }
            }
          }
        }
      }
    }
  }
}

#else

__global__ void __hashmult2(int nrows, int nfeats, int ncols, int brows1, int brows2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose) {}

#endif

int hashmult(int nrows, int nfeats, int ncols, int brows1, int brows2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose) {
  if (nrows >= 0) {
    int nt = max(1, 256/nrows);
    dim3 threadDim(nrows, nt, 1);
    int nblocks = min(MAXXGRID, ncols);
    __hashmult<<<nblocks,threadDim>>>(nrows, nfeats, ncols, brows1, brows2, A, Bdata, Bir, Bjc, C, transpose);
  } else {
    dim3 threadDim(32, 1, 1);
    int nblocks = min(MAXXGRID, ncols);
    __hashmult2<<<nblocks,threadDim>>>(nrows, nfeats, ncols, brows1, brows2, A, Bdata, Bir, Bjc, C, transpose);
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

//__forceinline__ __device__ long long __pairembed(long long r1, int r2) {
//  return ((r1+r2)*(r1+r2+1) >> 1) + r2;
//}

// The pair embedding function assumes r1x > r2x >= 0; 

__forceinline__ __device__ long long __pairembed(long long r1x, int r2x) {
  long long r1 = r1x+1;
  int r2 = r2x+1;
  float loc1 = (float) r1;
  float loc2 = (float) r2;
  int nbits1 = ((*(int *)(&loc1)) >> 23) - 126;
  int nbits2 = ((*(int *)(&loc2)) >> 23) - 126;
  int len = nbits1 + nbits2 - 2;
  float loc3 = (float) len; 
  int lenbits = 0;
  if (len > 1) lenbits = ((*(int *)(&loc3)) >> 23) - 127;
  r2 = r2 & ((1 << (nbits2-1)) - 1);
  long long x = (((r1 << (nbits2-1)) | r2) << lenbits) | (nbits2-1);
  return max((long long)0,x-2);
}

__global__ void __dopairembed(int *r1, int *r2, long long *res, int n) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    res[i] = __pairembed(r1[i], r2[i]);
  }
}

int pairembed(int *r1, int *r2, long long *res, int n) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __dopairembed<<<griddims,nthreads>>>(r1, r2, res, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


// Pair mult multplies base features and pairs of features.
//
// NOTE: The single-matrix version uses a fast lookup recurrence which is only valid up to 3000 base features per column (approx 4.5 million pairs)

// Given dense A and sparse B, for each column of B, enumerate all pairs of features, hash to a single feature index, and multiply by A into C
// todo: fix offsets

__global__ void __pairmult(int nrows, int bncols, int brows1, int brows2, float *A, int lda, float *A2, int lda2, 
                           float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, float *C, int ldc, int transpose) {
  bool doit = false;
  int istart = ((long long)blockIdx.x) * bncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * bncols / gridDim.x;
  float *AX;
  int ldax;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i + bcoff];                             // Range of nz rows in this column
    int jend = Bjc[i+1 + bcoff];
    int nr = jend - jstart;                                  // Number of nz rows
    int todo = nr * (nr + 1) / 2;                            // Number of pairs to process (including k,k pairs)
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = solve1(j);                                    // Compute the first and second indices
      int j2 = j - j1*(j1+1)/2; 
      //      int j1, j2;
      //      solvex(todo, j, j1, j2);
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1] - broff;                             // And their row indices
      int r2 = Bir[jstart + j2] - broff;
      long long rank = r1;
      float prod = f1;
      doit = (r1 >= 0 && r2 >= 0);
      if (j1 == j2) {
        doit = doit && r1 < brows1;
        AX = A;
        ldax = lda;
      } else {
        rank = __pairembed(r1, r2);
        doit = doit && (rank >= 0 && rank < brows2);
        if (doit) {
          prod = f1*f2/(abs(f1)+abs(f2)+1.0e-7f);
          AX = A2;
          ldax = lda2;
        }
      }
      if (doit) {
        if (transpose > 0) {
          float sum = AX[threadIdx.x + ldax * i] * prod;    // Do the product
          atomicAdd(&C[threadIdx.x + ldc * rank], sum);
        } else {
          float sum = AX[threadIdx.x + ldax * rank] * prod;  // Do the product
          atomicAdd(&C[threadIdx.x + ldc * i], sum);
        }
      }
    }
  }
}

#if __CUDA_ARCH__ >= 300

// This version is designed for few (or one) rows in A. It allocates one warp per column
// todo: implement the offsets. 

__global__ void __pairmult2(int nrows, int bncols, int brows1, int brows2, float *A, int lda, float *A2, int lda2, 
                            float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, float *C, int ldc, int transpose) {
  bool doit = false;
  int istart = ((long long)blockIdx.x) * bncols / gridDim.x;
  int iend = ((long long)(blockIdx.x+1)) * bncols / gridDim.x;
  float *AX;
  int ldax;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i + bcoff];                                     // Range of nz rows in this column
    int jend = Bjc[i+1 + bcoff];
    int nr = jend - jstart;                                  // Number of nz rows
    for (int j1 = 0; j1 < nr; j1 += blockDim.x) {               // work on a block of data
      float f1 = 0;
      int r1 = -1;
      if (j1 + threadIdx.x < nr) {
        f1 = Bdata[jstart + j1 + threadIdx.x];                // Get the two features
        r1 = Bir[jstart + j1 + threadIdx.x] - broff;                  // And their row indices
      }
      for (int j2 = j1; j2 < nr; j2 += blockDim.x) {             // work on a block of data
        float f2 = 0;
        int r2 = -1;
        if (j2 + threadIdx.x < nr) {
          f2 = Bdata[jstart + j2 + threadIdx.x];
          r2 = Bir[jstart + j2 + threadIdx.x] - broff;
        }
        for (int k = 0; k < 32; k++) {
          float f2shift = __shfl(f2, k);
          int r2shift = __shfl(r2, k);
          if (j2 + k < nr && r1 >= 0) {
            long long rank = r1;
            float prod = f1;
            doit = (r1 >= 0 && r1 < brows1 && r2 >= 0 && r2 < brows1);
            if (j1 + threadIdx.x == j2 + k) {
              AX = A;
              ldax = lda;
            } else if (j1 + threadIdx.x < j2 + k) {
              rank = __pairembed(r1, r2);
              doit = doit && (rank < brows2);
              if (doit) {
                prod *= f2shift;
                AX = A2;
                ldax = lda2;
              }
            }
            if (doit) {
              if (transpose > 0) {
                for (int m = 0; m < nrows; m++) {
                  float sum = AX[m + ldax * i] * prod;    // Do the product
                  atomicAdd(&C[m + ldc * rank], sum);
                  //		  atomicAdd(&C[0], sum);
                }
              } else {
                for (int m = 0; m < nrows; m++) {
                  float sum = AX[m + ldax * rank] * prod;  // Do the product
                  atomicAdd(&C[m + ldc * i], sum);
                  //		  atomicAdd(&C[0], sum);
                }
              }
            }
          }
        }
      }
    }
  }
}

#else

__global__ void __pairmult2(int nrows, int bncols, int brows1, int brows2, float *A, int lda, float *A2, int lda2, 
                            float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, float *C, int ldc, int transpose) {}

#endif

int pairMultTile(int nrows, int bncols, int brows1, int brows2, float *A, int lda, float *A2, int lda2, 
                 float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, float *C, int ldc, int transpose) {
  if (nrows >= 0) {
    int nt = max(1, 256/nrows);
    dim3 threadDim(nrows, nt, 1);
    int nblocks = min(MAXXGRID, bncols);
    __pairmult<<<nblocks,threadDim>>>(nrows, bncols, brows1, brows2, A, lda, A2, lda2, Bdata, Bir, Bjc, broff, bcoff, C, ldc, transpose);
  } else {
    dim3 threadDim(32, 1, 1);
    int nblocks = min(MAXXGRID, bncols);
    __pairmult2<<<nblocks,threadDim>>>(nrows, bncols, brows1, brows2, A, lda, A2, lda2, Bdata, Bir, Bjc, broff, bcoff, C, ldc, transpose);
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

__forceinline__ __device__ void __gupdate(float grad, int i, int ihere, int jhere, float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                                              float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  float lr, ve, te, pve, ste, ngrad, ssq, ssqnew;
  ssq = Sumsq[ihere];
  ssqnew = hypotf(grad,ssq);
  atomicAdd(&Sumsq[ihere], ssqnew - ssq);
  ssq = ssqnew * sqrtf(istep);

  if (addgrad) {
    lr =  (lrlen > 1) ? lrate[i] : lrate[0];
    ve =  (vexplen > 1) ? vexp[i] : vexp[0];
    te =  (texplen > 1) ? texp[i] : texp[0];
    pve = (ve == 0.5f) ? ssq : ((ve == 0) ? 1.0f : pow(ssq, 2*ve));
    ste = pow(istep, te);
    ngrad = grad * lr * ste / pve;
    atomicAdd(&MM[ihere], ngrad);
  }
  if (Mask != NULL) {
    if (maskrows > 1) {
      if (Mask[ihere] == 0) MM[ihere] = 0;
    } else {
      if (Mask[jhere] == 0) MM[ihere] = 0;
    }
  }
}

/*
__forceinline__ __device__ void __gupdate(float grad, int i, int ithere, int jthere, float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                                          float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  float lr, ve, te, pve, ste, ngrad;
  Sumsq[ithere] += grad * grad + epsilon;
  if (addgrad) {
    lr =  (lrlen > 1) ? lrate[i] : lrate[0];
    ve =  (vexplen > 1) ? vexp[i] : vexp[0];
    te =  (texplen > 1) ? texp[i] : texp[0];
    pve = (ve == 0) ? 1.0f : pow(Sumsq[ithere] * istep, ve);
    ste = pow(istep, te);
    ngrad = grad * lr * ste / pve;
    atomicAdd(&MM[ithere], ngrad);
  }
  if (Mask != NULL) {
    if (maskrows > 1) {
      if (Mask[ithere] == 0) MM[ithere] = 0;
    } else {
      if (Mask[jthere] == 0) MM[ithere] = 0;
    }
  }
  }*/

__global__ void __hashmultADAGrad(int nrows, int nfeats, int ncols, int brows1, int brows2, float *A, float *Bdata, int *Bir, int *Bjc, int transpose,
                                  float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                                  float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  bool doit = false;
  int ihere, ithere, jthere;
  float grad;
  int istart = ((long long)blockIdx.x) * ncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * ncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i];                                     // Range of nz rows in this column
    int jend = Bjc[i+1];
    int nr = jend - jstart;                                  // Number of nz rows
    int todo = nr * (nr + 1) / 2;                            // Number of pairs to process (including k,k pairs)
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = solve1(j);                                    // Compute the first and second indices
      int j2 = j - j1*(j1+1)/2; 
      //      int j1, j2;
      //      solvex(todo, j, j1, j2);
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1];                             // And their row indices
      int r2 = Bir[jstart + j2];
      long long rank = r1 + 1;
      float prod = f1;
      if (j1 == j2) {
        doit = (rank < brows1);
      } else {
        prod *= f2;
        rank *= r2 + 1;
        doit = (rank < brows2);
      }
      if (doit) {
        int ind = mmhash2(r1, r2, nfeats);                     // Hash the indices
        if (transpose > 0) {
          ihere = threadIdx.x + nrows * i;
          ithere = threadIdx.x + nrows * ind;
          jthere = ind;
        } else {
          ithere = threadIdx.x + nrows * i;
          jthere = i;
          ihere = threadIdx.x + nrows * ind;
        }
        grad = A[ihere] * prod;    // raw gradient
        __gupdate(grad, threadIdx.x, ithere, jthere, MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
      }
    }
  }
}

int hashmultADAGrad(int nrows, int nfeats, int ncols, int brows1, int brows2, float *A, float *Bdata, int *Bir, int *Bjc, int transpose, 
                    float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                    float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, ncols);
  __hashmultADAGrad<<<nblocks,threadDim>>>(nrows, nfeats, ncols, brows1, brows2, A, Bdata, Bir, Bjc, transpose, 
                                           MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}
//
// nrows = rows of MM (and other model mats)
// ncols = columns of B = columns of A 
//
__global__ void __pairMultADAGradTile(int nrows, int bncols, int brows1, int brows2, float *A, int lda, int aroff, int acoff, 
                                      float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, int transpose,
                                      float *MM, int ldmm, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                                      float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  bool doit = false;
  int ihere, ithere, jhere, jthere;
  float grad;
  int istart = ((long long)blockIdx.x) * bncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * bncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i+bcoff];                               // Range of nz rows in this column
    int jend = Bjc[i+1+bcoff];
    int nr = jend - jstart;                                  // Number of nz rows
    int todo = nr * (nr + 1) / 2;                            // Number of pairs to process (including k,k pairs)
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = solve1(j);                                    // Compute the first and second indices
      int j2 = j - j1*(j1+1)/2; 
      //      int j1, j2;
      //      solvex(todo, j, j1, j2);
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1]-broff;                       // And their row indices
      int r2 = Bir[jstart + j2]-broff;
      long long rank = r1;
      float prod = f1;
      doit = (r1 >= 0 && r2 >= 0);
      if (doit) {
        if (j1 == j2) {
          doit = doit && r1 < brows1;
          ithere = 0;
          jthere = 0;
        } else {
          rank = __pairembed(r1, r2);
          doit = doit && (rank < brows2);
          if (doit) {
            prod = f1*f2/(abs(f1)+abs(f2)+1.0e-7f);
            ithere = ldmm;
            jthere = 1;
          }
        }
      }
      if (doit) {
        if (transpose > 0) {
          ihere = threadIdx.x + aroff + lda * (i + acoff);
          jhere = threadIdx.x + aroff;
          ithere += threadIdx.x + 2 * ldmm * rank;
          jthere += 2 * rank;
        } else {
          ihere = threadIdx.x + aroff + lda * (rank + acoff);
          jhere = threadIdx.x + aroff;
          ithere += threadIdx.x + 2 * ldmm * i;
          jthere += 2 * i;
        }
        grad = A[ihere] * prod;    // raw gradient
        __gupdate(grad, jhere, ithere, jthere, MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
      }
    }
  }
}

int pairMultADAGradTile(int nrows, int bncols, int brows1, int brows2, float *A, int lda, int aroff, int acoff,
                        float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, int transpose, 
                        float *MM, int ldmm, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                        float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, bncols);
  __pairMultADAGradTile<<<nblocks,threadDim>>>(nrows, bncols, brows1, brows2, A, lda, aroff, acoff, Bdata, Bir, Bjc, broff, bcoff, transpose, 
                                               MM, ldmm, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void __hashcross(int nrows, int nfeats, int ncols,
			     float *A,
			     float *Bdata, int *Bir, int *Bjc,
			     float *Cdata, int *Cir, int *Cjc,
			     float *D, int transpose) {
  int r1, r2, ind;
  int istart = ((long long)blockIdx.x) * ncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * ncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart1 = Bjc[i];                                    // Range of nz rows in this column of B
    int jend1 = Bjc[i+1];
    int jstart2 = Cjc[i];                                    // Range of nz rows in this column of C
    int jend2 = Cjc[i+1];
    int nr1 = jend1 - jstart1;                               // Number of nz rows
    int nr2 = jend2 - jstart2;                               // Number of nz rows
    int todo = (nr1+1) * (nr2+1) - 1;                        // Number of pairs + singletons to process 
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = j / nr2;
      int j2 = j - j1 * nr2; 
      float prod = 1.0f;
      int hash = seed;
      if (j1 < nr1) {
        prod *= Bdata[jstart1 + j1];                         // Get the two features
        r1 = Bir[jstart1 + j1];                              // And their row indices
        hash = h1(r1, hash);
      }
      if (j2 < nr2) {
        prod *= Cdata[jstart2 + j2];
        r2 = Cir[jstart2 + j2];
        hash = h1(r2, hash);                                 // Hash the indices
      } 
      ind = mmhashend(hash, nfeats);
      if (transpose > 0) {
        float sum = A[threadIdx.x + nrows * i] * prod;       // Do the product
        atomicAdd(&D[threadIdx.x + nrows * ind], sum);
      } else {
        float sum = A[threadIdx.x + nrows * ind] * prod;     
        atomicAdd(&D[threadIdx.x + nrows * i], sum);
      }
    }
  }
}

int hashcross(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *Cdata, int *Cir, int *Cjc, float *D, int transpose) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, ncols);
  __hashcross<<<nblocks,threadDim>>>(nrows, nfeats, ncols, A, Bdata, Bir, Bjc, Cdata, Cir, Cjc, D, transpose);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}
