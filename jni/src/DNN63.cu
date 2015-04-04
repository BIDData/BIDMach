#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

#define WLENB 8
#define BYDIM 10

#if __CUDA_ARCH__ >= 300

/*
 *
 * Simple backward convolution kernel for word2vec. 
 * Computes the gradient for A given B or vice-versa, and does an SGD update.
 * 
 *  SKIP is the max skip-gram length
 *  WINLEN is the length of a block of columns to process
 *
 */


template<int NWA, int NWB, int AnotB>
  __global__ void __word2vecBwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C, float lrate) {
  const int nwab = NWA * NWB;
  __shared__ float cc[nwab];
  float aa[NWA];
  float bb[NWB];
  int wa[NWA];
  int wb[NWB];
  int tid = threadIdx.x;
  int fid = threadIdx.x + blockDim.x * threadIdx.y;
  int dxy = blockDim.x * blockDim.y;
  int icol, i, j, k;
  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

  for (icol = istart; icol < iend; icol++) {            // iterate in columns
    __syncthreads();
    for (i = fid; i < nwab; i += dxy) {                 // Load C
      cc[i] = C[i + icol * nwab];
    }
    __syncthreads();
    for (i = tid; i < nrows; i += blockDim.x) {
      if (AnotB) {
#pragma unroll
        for (j = 0; j < NWA; j++) {                       // Clear A, the output vector;
          aa[j] = 0;
        }
      } else {
#pragma unroll
        for (j = 0; j < NWB; j++) {                       // Clear B, the output vector;
          bb[j] = 0;      
        }
      }

#pragma unroll
      for (j = 0; j < NWA; j++) {
        wa[j] = WA[j + icol * NWA];                       // Load the A word matrix
    }
#pragma unroll
      for (j = 0; j < NWB; j++) {
        wb[j] = WB[j + icol * NWB];                       // Load the B word matrix
      }

      if (AnotB) {
#pragma unroll
        for (j = 0; j < NWB; j++) {                       // Load the data
          bb[j] = B[i + wb[j] * nrows];
        }
#pragma unroll
        for (j = 0; j < NWA; j++) {                       // Load the data
          aa[j] = A[i + wa[j] * nrows];
        }
      }

#pragma unroll
      for (j = 0; j < NWA; j++) {                         // Now do the product
#pragma unroll
        for (k = 0; k < NWB; k++) {                       
          if (AnotB) {
            aa[j] += cc[j + k * NWA] * bb[k];
          } else {
            bb[k] += cc[j + k * NWA] * aa[j];
          }
        }
      }

      if (AnotB) {
#pragma unroll
        for (j = 0; j < NWA; j++) {                         // Output the product
          atomicAdd(&A[i + wa[j] * nrows], aa[j] * lrate);
        }
      } else {
#pragma unroll
        for (j = 0; j < NWB; j++) {                         // Output the product
          atomicAdd(&B[i + wb[j] * nrows], bb[j] * lrate);
        }
      }
    }
  }  
}

/*
 *
 * Simple backward convolution kernel for word2vec. 
 * Computes the gradient for A given B or vice-versa, and does an SGD update.
 * 
 *  SKIP is the max skip-gram length
 *  WINLEN is the length of a block of columns to process
 *
 */


template<int SKIP, int WINLEN, int HEIGHT, int AnotB>
  __global__ void __word2vecBwdx(int nrows, int ncols, int *W, float *A, float *B, float *C, float lrate) {
  const int window = 2*SKIP+1;
  __shared__ float cc[(WINLEN + 2*SKIP) * window];
  float bb[WINLEN + 2*SKIP];
  int word[WINLEN + 2*SKIP];
  int tid = threadIdx.x;
  int fid = threadIdx.x + blockDim.x * threadIdx.z;
  int icol, i, j;
  float sum;
  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

#pragma unroll
  for (i = 0; i < 2 * SKIP; i++) {                            // init context words on edges
    if (i + istart - SKIP > 0) {
      word[i + WINLEN] = W[i + istart - SKIP];
    } else {
      word[i + WINLEN] = 0;
    }
    if (AnotB) {                                              // Flag that sets whether updating A vs B
      bb[i + WINLEN] = B[tid + word[i + WINLEN] * nrows];     // Load edge data for the input matrix
    } else {
      bb[i + WINLEN] = A[tid + word[i + WINLEN] * nrows];
    }
  }

  __syncthreads();
  if (fid < window * 2 * SKIP) {
    if (fid + (istart - SKIP) * window > 0) {                 // Init the C block
      cc[fid + WINLEN * window] = C[fid + (istart - SKIP) * window];
    } else {
      cc[fid + WINLEN * window] = 0;
    }
  }
  __syncthreads();

  for (icol = istart; icol < iend; icol += WINLEN ) {         // iterate in blocks of WINLEN columns
#pragma unroll
    for (i = 0; i < 2*SKIP; i++) {                            // Shift the edges of the previous data block
      word[i] = word[i + WINLEN];
      bb[i] = bb[i + WINLEN];
    }
    __syncthreads();
    if (fid < window * 2 * SKIP) {                            // Shift the edges of the C block
      cc[fid] = cc[fid + WINLEN * window];
    }
    __syncthreads();
#pragma unroll
    for (i = 0; i < WINLEN; i++) {                            // Get new word and B and C matrix data
      if (i + icol + 2*SKIP < ncols) {
        word[i + 2*SKIP] = W[i + icol + 2*SKIP];
      } else {
        word[i + 2*SKIP] = 0;
      }
      if (tid < nrows) {
        if (AnotB) {                                          // Flag that sets whether updating A vs B
          bb[i + 2*SKIP] = B[tid + word[i + 2*SKIP] * nrows]; // Get new data aligned with the word vector
        } else {
          bb[i + 2*SKIP] = A[tid + word[i + 2*SKIP] * nrows];
        }
      }
      // Get new C data
      if (fid < window * WINLEN && fid + (icol+2*SKIP) * window < ncols * window) {
        cc[fid + 2*SKIP*window] = C[fid + (icol+2*SKIP) * window];
      }
    } 
    __syncthreads();
#pragma unroll
    for (i = 0; i < WINLEN; i++) {                            // Now compute products of B and C in this row
      sum = 0;
#pragma unroll
      for (j = 0; j <= 2 * SKIP; j++) {
        if (AnotB) {                                          // Now compute products of B and C in this row
          if (tid < nrows && i + icol < ncols) {
            sum += bb[i + j] * cc[(2*SKIP - j) + window * (i + j)];
          }
        } else {
          if (tid < nrows && i + icol < ncols) {
            sum += bb[i + j] * cc[j + window * (i + SKIP)];
          }
        }
      }
      if (tid < nrows && i + icol < ncols) {
        if (AnotB) {
          atomicAdd(&A[tid + word[i] * nrows], sum * lrate);
        } else {
          atomicAdd(&B[tid + word[i] * nrows], sum * lrate);
        }
      }
    }
  }  
}


#else

template<int NWA, int NWB, int AnotB>
  __global__ void __word2vecBwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C, float lrate) {}


#endif

int word2vecBwd(int nrows, int ncols, int *WA, int nwa, int *WB, int nwb, float *A, float *B, float *C, float lrate, int AnotB) {
  dim3 threads(32*BYDIM, 1, 1);
  int nblocks = min(2048, 2 + (ncols - 1)/WLENB);
  if (AnotB > 0) {
    __word2vecBwd<5,11,1><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate);
  } else {
    __word2vecBwd<5,11,0><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate);
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}
