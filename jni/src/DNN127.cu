#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>


#define NTZ 8
#define NTB (32/NTZ)

#define WLEN 6
#define WLENB 16
#define BYDIM 2

#if __CUDA_ARCH__ >= 300

/*
 *
 * Simple forward convolution kernel for word2vec. Computes the inner products of each column of A with a nearby column of B. 
 * 
 *  SKIP is the max skip-gram length
 *  WINLEN is the length of a block of columns to process
 *
 *  Columns of the output matrix C are <window> (2*SKIP+1) long, contain inner products with corresponding columns of B. 
 *  the row index of C specifies an offset from -SKIP ... SKIP into A, which is the column used for the inner product.
 *  i.e. C(i,j) = <B(:,j), A(:,j-SKIP+i)>
 *
 */

template<int SKIP, int WINLEN, int BDIM>
__global__ void __word2vecFwd(int nrows, int ncols, int *W, float *A, float *B, float *C) {
  const int window = 2*SKIP+1;
  float aa[WINLEN + 2*SKIP];
  float bb[WINLEN];
  float prods[WINLEN][window];
  int word[WINLEN + 2*SKIP];
  __shared__ float CC[WINLEN*BDIM*window];
  int i, j, k, icol;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int dxy = blockDim.x * blockDim.y;
  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

#pragma unroll
  for (i = 0; i < 2*SKIP; i++) {                           // init context words on edges
    if (i + istart - SKIP > 0) {
      word[i + WINLEN] = W[i + istart - SKIP];
    }
  }

  for (icol = istart; icol < iend; icol += WINLEN) {        // Iterate over columns in blocks of WINLEN
#pragma unroll
    for (j = 0; j < 2*SKIP; j++) {                         // Shift edge words from last time
      word[j] = word[j + WINLEN];
    }
#pragma unroll
    for (i = 0; i < WINLEN; i++) {                          // Get the new words in this window
      if (i + icol + 2*SKIP < ncols) {
        word[j + 2*SKIP] = W[i + icol + 2*SKIP];
      } else {
        word[i + 2*SKIP] = 0;
      }
#pragma unroll
      for (j = 0; j <= 2*SKIP; j++) {                      // clear the products matrix
        prods[i][j] = 0;
      }
    }

    for (i = tid; i < nrows; i += dxy) {                    // Now iterate over the rows of this block
#pragma unroll
      for (j = 0; j < WINLEN + 2*SKIP ; j++) {             // Read A with edges
        aa[j] = A[i + word[j] * nrows];
      }
#pragma unroll
      for (j = 0; j < WINLEN ; j++) {                       // Read B w/o edges, offset by SKIP
        bb[j] = B[i + word[j + SKIP] * nrows];
      }
#pragma unroll
      for (j = 0; j < WINLEN; j++) {                        // Computes the products of these elements
#pragma unroll
        for (k = 0; k <= 2*SKIP; k++) {
          prods[j][k] += aa[j+k] * bb[j];
        }
      }
    }                                                       // Finished the entire block

#pragma unroll
    for (i = 0; i < WINLEN; i++) {                          // Reduce the products within each warp
#pragma unroll
      for (j = 0; j <= 2*SKIP; j++) {
#pragma unroll
        for (k = 1; k < 32; k = k+k) {
          float tmp = __shfl_down(prods[i][j], k);
          prods[i][j] += tmp;
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {                                 // Save the products to SHMEM (one copy per warp)
#pragma unroll
      for (j = 0; j < WINLEN; j++) {
#pragma unroll
        for (k = 0; k < window; k++) {
          CC[k + window * (j + WINLEN * threadIdx.y)] = prods[j][k];
        }
      }
    }

    __syncthreads();
    for (j = 0; j < WINLEN * window; j += dxy) {            // Reduce the products across warps
      for (k = 1; k < blockDim.y; k++) {
        __syncthreads();
        if (j + tid < WINLEN * window) {
          CC[j + tid] += CC[j + tid + k * WINLEN * window];
        }
      } 
      __syncthreads();
      if (j + tid < WINLEN * window && j + tid + icol * window < iend * window) {
        C[j + tid + icol * window] = CC[j + tid];   // Save the results
      }
    }
    __syncthreads();
  }
}

// Custom convolution kernel for vectors

template<int SKIP>
__global__ void __convRows(int nrows, int ncols, float *A, int lda, float *B, int ldb, float *C) {
  const int window = 2*SKIP+1; 
  const int height = 32 - 2 * SKIP;
  float prods[window];
  int i, j, k;
  int gid = threadIdx.y + blockDim.y * blockIdx.x;
  int tid = threadIdx.x + height * gid;
  float a, b;
  if (tid < nrows) {
#pragma unroll
    for (k = 0; k < window; k++) {
      prods[k] = 0;
    }
    for (i = 0; i < ncols; i++) {
      a = A[tid + i*lda];
      b = B[tid + i*ldb];
#pragma unroll
      for (j = 0; j < height; j++) {
#pragma unroll
        for (k = -SKIP; k <= 0; k++) {
          prods[k+SKIP] += a * __shfl_up(b, -k);
        }
#pragma unroll
        for (k = 1; k <= SKIP; k++) {
          prods[k+SKIP] += a * __shfl_down(b, k);
        }
      }
    }
    if (threadIdx.x >= SKIP && threadIdx.x < 32-SKIP) {
#pragma unroll
      for (k = 0; k < window; k++) {
        C[k + window * (tid - SKIP)] = prods[k];
      }
    }
  }
}

template<int SKIP>
__global__ void __convColsx(int nrows, int ncols, int *W, float *A, float *B, float *C) {
  const int window = 2*SKIP+1; 
  const int width = 32 - 2*SKIP;
  __shared__ float AA[width][33];
  __shared__ float BB[32][33];
  __shared__ float CC[window][33];
  __shared__ int WW[32];
  float prods[window];

  int i, j, k, tid, fid, jcol, word, dxy;
  float a, b;
  dxy = blockDim.x * blockDim.y;
  tid = threadIdx.x + blockDim.x * threadIdx.y;
  fid = tid + dxy * threadIdx.z;
  __syncthreads();                                 
  for (jcol = width * blockIdx.x; jcol < ncols; jcol += width * gridDim.x) {

    __syncthreads();                                 
    if (tid + jcol < ncols) {                             // Load the words for this chunk
      WW[tid] = W[tid + jcol];
    } else {
      WW[tid] = 0;
    }
    __syncthreads();                           

    for (j = threadIdx.z; j < window; j+= blockDim.z) {   // Clear the shared product store
      CC[j][tid] = 0;
    }

    __syncthreads();
#pragma unroll                 
    for (k = 0; k < window; k++) {                        // Clear the register product store
      prods[k] = 0;
    }
    for (i = 0; i < nrows; i += dxy) {                    // process a block of this column
      __syncthreads();
      for (j = threadIdx.z; j < dxy; j += blockDim.z) {   // load data into SHMEM
        word = WW[j];
        if (i + tid < nrows) {
          if (j >= SKIP && j < dxy - SKIP) {
            AA[j-SKIP][tid] = A[i + tid + word * nrows];
          }
          BB[j][tid] = B[i + tid + word * nrows];
        }
      }
      __syncthreads();
#pragma unroll
      for (j = 0; j < NTB; j++) {                         // Get some SHMEM data into registers
        if (tid < width) {
          a = AA[tid][j + NTB*threadIdx.z];
        }
        b = BB[tid][j + NTB*threadIdx.z];
#pragma unroll
        for (k = 0; k < window; k++) {                    // compute shifted products
          prods[k] += a * __shfl_down(b, k);
        }
      }
      __syncthreads();
    }
  
    if (fid < 32) {
#pragma unroll
      for (k = 0; k < window; k++) {                      // move shifted products to SHMEM
        CC[k][tid] = prods[k];
      }
    }
    __syncthreads();
    if (fid >= 32) {
#pragma unroll
      for (k = 0; k < window; k++) {                      // move shifted products to SHMEM
        atomicAdd(&CC[k][tid], prods[k]);
      }
    }
    __syncthreads();                                      // save out to main memory
    if (tid + jcol < ncols) {
      for (i = threadIdx.z; i < window; i += blockDim.z) {
        C[i + (tid + jcol) * window] = CC[i][tid];
      }
    }
    __syncthreads();  
  }
}

#else

template<int SKIP>
__global__ void __convRows(int nrows, int ncols, float *A, int lda, float *B, int ldb, float *C) {}

template<int SKIP, int WINLEN, int BDIM>
__global__ void __word2vecFwd(int nrows, int ncols, int *W, float *A, float *B, float *C) {}


#endif

int convRows(int nrows, int ncols, int shift, float *A, int lda, float *B, int ldb, float *C) {
  dim3 threads(32, 32, 1);
  int nblocks = 1 + (nrows - 1)/threads.y;
  switch(shift) {
  case 5 : __convRows<5><<<nblocks,threads>>>(nrows, ncols, A, lda, B, ldb, C); break;
  case 10 : __convRows<10><<<nblocks,threads>>>(nrows, ncols, A, lda, B, ldb, C); break;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}

int word2vecFwd(int nrows, int ncols, int shift, int *W, float *A, float *B, float *C) {
  dim3 threads(32, BYDIM, 1);
  int nblocks = min(4*2048, 2 + (ncols - 1)/WLEN);
  switch(shift) {
  case 5 : __word2vecFwd<5,WLEN,BYDIM><<<nblocks,threads>>>(nrows, ncols, W, A, B, C); break;
    //  case 10 : __convCols<7><<<nblocks,threads>>>(nrows, ncols, W, A, B, C); break;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}

