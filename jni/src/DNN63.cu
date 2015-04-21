#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

#define CDIM 5

#if __CUDA_ARCH__ >= 300

/*
 * Convolutional kernel for word2vec. This handles the positively-label word pairs with
 * one context word and the current word. 
 */

template<int SKIP, int YDIM, int NREPS>
  __global__ void __word2vecPos(int nrows, int ncols, int *W, int *LB, int *UB, float *A, float *B, float lrate) {
  const int nwindow = 2*SKIP+1; 
  int words[nwindow];
  float aa[NREPS][nwindow];
  float daa[NREPS][nwindow];
  float bb[NREPS];
  float dbb[NREPS];
  __shared__ float CC[YDIM * nwindow];

  int i, j, k, tid, indx, icol, dxy, lb, ub;
  float prod, v;
  tid = threadIdx.x + blockDim.x * threadIdx.y;
  dxy = blockDim.x * blockDim.y;
  bool good;

  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

#pragma unroll
  for (i = 0; i < nwindow; i++) {                           // Prefill the word and aa window buffers
    if (istart + i - SKIP - 1 > 0) {
      words[i] = nrows * W[istart + i - SKIP - 1];          // Get a new word
    } else {
      words[i] = -1;
    }
    good = (words[i] >= 0);
#pragma unroll
    for (j = 0; j < NREPS; j++) {                           // Get the A vector for this word
      indx = tid + j * dxy;
      if (good && indx < nrows) {
        aa[j][i] = A[indx + words[i]];
      } else {
        aa[j][i] = 0;
      }
      daa[j][i] = 0;
    }
  }

  for (icol = istart; icol < iend; icol++) {                // Iterate over columns
#pragma unroll
    for (i = 0; i < nwindow-1; i++) {                       // slide words down
      words[i] = words[i+1];
#pragma unroll
      for (j = 0; j < NREPS; j++) {
        aa[j][i] = aa[j][i+1];                              // slide data down
        daa[j][i] = daa[j][i+1];                            // slide deriv down
      }
    }

    good = (icol + SKIP < ncols);
    if (good) {
      words[nwindow - 1] = nrows * W[icol + SKIP];          // Get a new word
    } else {
      words[nwindow - 1] = -1;
    }
    good = good && words[nwindow-1] >= 0;

#pragma unroll
    for (j = 0; j < NREPS; j++) {                           // Get a new A column
      indx = tid + j * dxy;
      if (good && indx < nrows) {
        aa[j][nwindow - 1] = A[indx + words[nwindow - 1]];
      } else {
        aa[j][nwindow - 1] = 0;
      }
      if (words[SKIP] >= 0 && indx < nrows) {               // Get a new B column
        bb[j] = B[indx + words[SKIP]];
      } else {
        bb[j] = 0;
      }
    }
    __syncthreads();
    
    lb = LB[icol];
    ub = UB[icol];

#pragma unroll                 
    for (i = 0; i < nwindow; i++) {                         // Iterate across the window for A cols
      prod = 0;
      if (i >= SKIP + lb && i <= SKIP + ub) {
#pragma unroll                 
        for (j = 0; j < NREPS; j++) {                       // Iterate over blocks of elements
          prod += aa[j][i] * bb[j];                         // Compute the product between current A, B cols
        }
#pragma unroll                 
        for (k = 1; k < 32; k = k + k) {
          prod += __shfl_down(prod, k);                     // Reduce within warp
        }  
        if (threadIdx.x == 0) {
          CC[i - SKIP -lb + threadIdx.y * nwindow] = prod;  // Save to SHMEM
        }
      }
    }

    __syncthreads();
    for (i = 0; i < blockDim.y; i++) {                      // Reduce across warps
      for (k = tid; k <= ub - lb; k += dxy) { 
        CC[k] = CC[k + i * nwindow];
      }
      __syncthreads();
    }

    __syncthreads();                                        //  Apply the sigmoid map
    for (i = tid; i <= ub - lb; i += dxy) { 
      v = CC[i];
      if (v > 16.0f) {
        v = 1.0f;
      } else {
        v = exp(v);
        v = v / (1.0f + v);
      }
      CC[i] = 1.0f - v;                                     // All pairs have label 1
    }
      
    __syncthreads();  
#pragma unroll                 
    for (j = 0; j < NREPS; j++) {
      dbb[j] = 0;
    }
#pragma unroll                 
    for (i = 0; i < nwindow; i++) {                         // Iterate across the window for A cols
      if (i >= SKIP + lb && i <= SKIP + ub) {
        v = lrate * CC[i - SKIP - lb];
#pragma unroll                 
        for (j = 0; j < NREPS; j++) {
          dbb[j] += v * aa[j][i];
          daa[j][i] += v * bb[j];                           // Compute the product with the current A, B cols
        }
      }
    }
#pragma unroll                 
    for (j = 0; j < NREPS; j++) { 
      if (words[SKIP] >= 0 && tid + j * dxy < nrows) {      // Save the B column
        atomicAdd(&B[tid + j * dxy + words[SKIP]], dbb[j]);
      }
    }
    __syncthreads();  
    if (icol - SKIP >= 0 && words[0] >= 0) {
      for (j = 0; j < NREPS; j++) {                         // Save the A column
        if (tid + j * dxy < nrows) {
          atomicAdd(&A[tid + j * dxy + words[0]], daa[j][0]);
        }
      } 
    }
  }
}

/*
 * Convolutional kernel for word2vec. This handles the positively-label word pairs with
 * one context word and the current word. 
 */

template<int SKIP, int YDIM, int NREPS>
  __global__ void __word2vecPos_exp(int nrows, int ncols, int *W, int *LB, int *UB, float *A, float *B, float lrate) {
  const int nwindow = 2*SKIP+1; 
  float aa[NREPS];
  float da[NREPS];
  __shared__ float CC[YDIM * nwindow];

  int i, j, k, tid, icol, dxy, lb, ub, iword, cword;
  float bb, db, prod, v;
  tid = threadIdx.x + blockDim.x * threadIdx.y;
  dxy = blockDim.x * blockDim.y;

  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

  for (icol = istart; icol < iend; icol++) {                // Iterate over columns

    iword = nrows * W[icol];                                // Get the current word

    __syncthreads();
    
    lb = LB[icol];
    ub = UB[icol];

    if (iword >= 0) {
#pragma unroll                 
      for (j = 0; j < NREPS; j++) {                         // Iterate over blocks of elements
        if (tid + j * dxy < nrows) {                        // Get A
          aa[j] = A[tid + j * dxy + iword];
        } else {
          aa[j] = 0;
        }
      }

      for (i = lb; i <= ub; i++) {                          // Iterate across the window for A cols
        __syncthreads();
        cword = nrows * W[icol + i];                        // Get the current word
        prod = 0;
        if (cword >= 0) {
#pragma unroll                 
          for (j = 0; j < NREPS; j++) {                     // Iterate over blocks of elements
            if (tid + j * dxy < nrows) {                    // Get B col
              bb = B[tid + j * dxy + cword];
              prod += aa[j] * bb;                           // Compute the product between current A, B cols
            }
          }
#pragma unroll                 
          for (k = 1; k < 32; k = k + k) {
            prod += __shfl_down(prod, k);                   // Reduce within warp
          }  
        }
        if (threadIdx.x == 0) {
          CC[i - lb + threadIdx.y * nwindow] = prod;        // Save to SHMEM
        }
      }

      __syncthreads();
      for (j = 1; j < blockDim.y; j++) {                    // Reduce across warps
        for (i = tid; i < ub - lb; i += dxy) { 
          CC[i] += CC[i + j * nwindow];
        }
        __syncthreads();
      }

      __syncthreads();                                      //  Apply the sigmoid map
      for (i = tid; i < ub - lb; i += dxy) { 
        v = CC[i];
        if (v > 16.0f) {
          v = 1.0f;
        } else {
          v = exp(v);
          v = v / (1.0f + v);
        }
        CC[i] = lrate * (1.0f - v);                         // All pairs have label 1
      }
      
      __syncthreads();  
#pragma unroll                 
      for (j = 0; j < NREPS; j++) {
        da[j] = 0;
      }
      for (i = lb; i <= ub; i++) {                          // Iterate across the window for A cols   
        cword = nrows * W[icol + i];                        // Get the context word
        v = CC[i - lb];
        if (cword >= 0) {
#pragma unroll                 
          for (j = 0; j < NREPS; j++) {                     // Iterate over blocks of elements
            if (tid + j * dxy < nrows) {                    // Get B col
              bb = B[tid + j * dxy + cword];
              da[j] += v * bb;
              db = v * aa[j];
              atomicAdd(&B[tid + j * dxy + cword], db);
            }
          }
        }
      }

#pragma unroll                 
      for (j = 0; j < NREPS; j++) {
        if (tid + j * dxy < nrows) {                    
          atomicAdd(&A[tid + j * dxy + iword], da[j]);
        }
      }
    }
  }
}


#else

template<int SKIP, int YDIM, int NREPS>
  __global__ void __word2vecPos(int nrows, int ncols, int *W, int *LB, int *UB, float *A, float *B, float lrate) {}

#endif

int word2vecPos(int nrows, int ncols, int skip, int *W, int *LB, int *UB, float *A, float *B, float lrate) {
  dim3 threads(32, CDIM, 1);
  int nblocks = 1 + (nrows - 1)/threads.y;
  switch(skip) {
  case 5 : __word2vecPos<5, CDIM, 10/CDIM><<<nblocks,threads>>>(nrows, ncols, W, LB, UB, A, B, lrate); break;
  case 3 : __word2vecPos<3, CDIM, 10/CDIM><<<nblocks,threads>>>(nrows, ncols, W, LB, UB, A, B, lrate); break;
  case 2 : __word2vecPos<2, CDIM, 10/CDIM><<<nblocks,threads>>>(nrows, ncols, W, LB, UB, A, B, lrate); break;
  default : printf("word2vecPos unsupport size %d\n", skip); return 1;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}
