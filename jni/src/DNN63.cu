#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

#define WLENB 8
#define BYDIM 5

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


template<int NWA, int NWB, int MAXDIM>
  __global__ void __word2vecBwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C, float lrate) {
  const int nwab = NWA * NWB;
  float dd[MAXDIM];
  int wa[NWA];
  int wb[NWB];
  __shared__ float cc[NWA*NWB];
  int tid = threadIdx.x;
  int fid = threadIdx.x + blockDim.x * threadIdx.y; 
  int dxy = blockDim.x * blockDim.y;
  int icol, i, j, k;
  float sum;
  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

  for (icol = istart; icol < iend; icol++) {            // iterate in columns
#pragma unroll
    for (j = 0; j < NWA; j++) {
      wa[j] = WA[j + icol * NWA];                       // Load the A word matrix
    }
    __syncthreads();
#pragma unroll 
    for (j = 0; j < NWB; j++) {
      wb[j] = WB[j + icol * NWB];                       // Load the B word matrix
    }
    for (i = fid; i < nwab; i += dxy) {
      cc[i] = C[i + icol * nwab];
    }
    __syncthreads();
    for (i = tid; i < nrows; i += dxy) {
#pragma unroll
      for (j = 0; j < NWB; j++) {                       // Load the data
        dd[j] = B[i + wb[j] * nrows];
      }

      for (j = 0; j < NWA; j++) {                         // Now do the product
        sum = 0;
#pragma unroll
        for (k = 0; k < NWB; k++) {                       
          float xx =  cc[j + k * NWA];
          sum += xx * dd[k];
        }
        atomicAdd(&A[i + wa[j] * nrows], sum * lrate);
      }

#pragma unroll
      for (j = 0; j < NWA; j++) {                       // Load the data
        dd[j] = A[i + wa[j] * nrows];
      }

      for (j = 0; j < NWB; j++) {                         // Now do the product
        sum = 0;
#pragma unroll
        for (k = 0; k < NWA; k++) {                       
          float xx =  cc[k + j * NWA];
          sum += xx * dd[k];
        }
        atomicAdd(&B[i + wb[j] * nrows], sum * lrate);
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

/*
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
*/

#else 

template<int NWA, int NWB, int MAXDIM>
  __global__ void __word2vecBwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C, float lrate) {}


#endif

int word2vecBwd(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float *C, float lrate) {
  dim3 threads(32*BYDIM, 1, 1);
  int nblocks = min(2048, 2 + (ncols - 1)/WLENB);
  int which = nwa*10000 + nwb;
  switch (which) {
  case 10005: __word2vecBwd<1,5,5><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 50005: __word2vecBwd<5,5,5><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 110005: __word2vecBwd<11,5,11><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 80006: __word2vecBwd<8,6,8><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}
