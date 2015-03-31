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


__forceinline__ __device__ int solve1(int j) {
  float v = sqrtf((float)j);
#pragma unroll
  for (int k = 0; k < 5; k++) {
    v = v - (v*(v+1)-2*j)/(2*v+1);   // Newton iterations to find first index. 
  }
  return (int)(v+2e-5f);   
}

// Feature hashing multiply and multiply-transpose.
// This one enumerates, hashes and multiplies all pairs of features.
//
// NOTE: The single-matrix version (hashmult) uses a fast lookup recurrence which is only valid up to 3000 base features per column (approx 4.5 million pairs)

// Given dense A and sparse B, for each column of B, enumerate all pairs of features, hash to a single feature index, and multiply by A into C

__global__ void __hashmult(int nrows, int nfeats, int ncols, int bound1, int bound2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose) {
  bool doit = false;
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
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1];                             // And their row indices
      int r2 = Bir[jstart + j2];
      int ind = mmhash2(r1, r2, nfeats);                     // Hash the indices
      long long rank = r1 + 1;
      float prod = f1;
      if (j1 == j2) {
        doit = (rank < bound1);
      } else {
        prod *= f2;
	rank *= r2 + 1;
        doit = (rank < bound2);
      }
      if (doit) {
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



int hashmult(int nrows, int nfeats, int ncols, int bound1, int bound2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, ncols);
  __hashmult<<<nblocks,threadDim>>>(nrows, nfeats, ncols, bound1, bound2, A, Bdata, Bir, Bjc, C, transpose);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

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
}

__global__ void __hashmultADAGrad(int nrows, int nfeats, int ncols, int bound1, int bound2, float *A, float *Bdata, int *Bir, int *Bjc, int transpose,
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
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1];                             // And their row indices
      int r2 = Bir[jstart + j2];
      int ind = mmhash2(r1, r2, nfeats);                     // Hash the indices
      long long rank = r1 + 1;
      float prod = f1;
      if (j1 == j2) {
        doit = (rank < bound1);
      } else {
        prod *= f2;
        rank *= r2 + 1;
        doit = (rank < bound2);
      }
      if (doit) {
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

int hashmultADAGrad(int nrows, int nfeats, int ncols, int bound1, int bound2, float *A, float *Bdata, int *Bir, int *Bjc, int transpose, 
                    float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                    float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, ncols);
  __hashmultADAGrad<<<nblocks,threadDim>>>(nrows, nfeats, ncols, bound1, bound2, A, Bdata, Bir, Bjc, transpose, 
                                           MM, Sumsq, Mask, maskrows, lrate, lrlen, vexp, vexplen, texp, texplen, istep, addgrad, epsilon);
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

