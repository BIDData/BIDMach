#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>


#define NTZ 8
#define NTB (32/NTZ)

#define WLEN 6
#define BYDIM 2

/*

#if __CUDA_ARCH__ >= 300

// nwords should be a multiple of NPOS
// shift is offset from current word to context word. 

template<int NPOS, int NNEG>
__global__ void __word2vecBlock(int nrows, int ncols, int nwords, int shift, int *W, float *D, float *C, float lrate, curandState *rstates) {
  float prods[NPOS][NNEG];
  float cprods[NPOS];
  float cval[NPOS];
  float dval[NPOS];
  float nval[NNEG];
  int word[NPOS];
  int cword[NPOS];
  __shared__ int nword[NNEG];
  __shared__ float sprods[NPOS * NNEG];
  __shared__ float scprods[NPOS];

  __shared__ float cderiv[NPOS];
  __shared__ float dderiv[NPOS];
  __shared__ float ndderiv[NNEG];

  int j, k, m, iword;
  float tmp;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int gid = blockIdx.x;
  curandState *rstate;
  if (tid < NNEG) {
    rstate = &rstates[tid + NNEG * gid];
  }
  int distance = ncols / NPOS;
  int shift0 = max(0, -shift);
  int shift1 = max(0, shift);

  for (iword = gid; iword < distance; iword += gridDim.x) {
    __syncthreads();
#pragma unroll
    for (j = 0; j < NPOS; j++) {                               // Get current and context words
      word[j] = W[iword + shift0 + j * distance];
      cword[j] = W[iword + shift1 + j * distance];
    }
    __syncthreads();
    if (tid < NNEG) {
      nword[tid] = min(nwords-1, (int)(nwords*curand_uniform(rstate)));
    }
    __syncthreads();

    // Clear product accumulators
#pragma unroll
    for (j = 0; j < NPOS; j++) {    
      cprods[j] = 0;
#pragma unroll
      for (k = 0; k < NNEG; k++) {    
        prods[j][k] = 0;
      }
    }

    // Now accumulate over each column
    if (tid < nrows) {                                            // Row index of the inner product
#pragma unroll
      for (j = 0; j < NPOS; j++) {    
        cval[j] = C[tid + cword[j]*nrows];
        dval[j] = D[tid + word[j]*nrows];
        cprods[j] += cval[j] * dval[j];
      }
#pragma unroll
      for (k = 0; k < NNEG; k++) {    
        nval[k] = D[tid + nword[k]*nrows];
#pragma unroll
        for (j = 0; j < NPOS; j++) {    
          prods[j][k] += nval[k] * cval[j];
        }
      }
    }   // done with this column

    // Reduce within warp
#pragma unroll
    for (j = 0; j < NPOS; j++) {    
#pragma unroll
      for (m = 1; m < 32; m = m + m) {
        tmp = __shfl_down(cprods[j], m);
        if (threadIdx.x >= m) {
          cprods[j] += tmp;
        }
      }
#pragma unroll
      for (k = 0; k < NNEG; k++) {    
#pragma unroll
        for (m = 1; m < 32; m = m + m) {
          tmp = __shfl_down(prods[j][k], m);
          if (threadIdx.x >= m) {
            prods[j][k] += tmp;
          }
        }
      }
    }
    // clear SHMEM
    __syncthreads();
    if (tid < NPOS) {
      scprods[tid] = 0;
    }
    if (tid < NPOS * NNEG) {
      sprods[tid] = 0;
    }
    __syncthreads();
    // accum into SHMEM
    if (threadIdx.x == 0) {
#pragma unroll
      for (j = 0; j < NPOS; j++) {    
        //        atomicAdd(&scprods[j], cprods[j]);
#pragma unroll
        for (k = 0; k < NNEG; k++) {    
          //          atomicAdd(&sprods[j + NPOS * k], prods[j][k]);
        }
      }
    }
    __syncthreads();

    // derivative magic here
    float sum = 0.0f;
#pragma unroll
    for (j = 0; j < NPOS; j++) {    
#pragma unroll
      for (k = 0; k < NNEG; k++) {    
        sum += prods[j][k] * cprods[j];
      }
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
      for (j = 0; j < NPOS; j++) {    
        cderiv[j] = 1.0e-5f * sum;
        dderiv[j] = 1.0e-5f * sum;
      }
#pragma unroll
      for (k = 0; k < NNEG; k++) {    
        ndderiv[k] = 1.0e-5f * sum;
      }
    }
    __syncthreads();

    // update the outputs
    if (tid < nrows) {                     
#pragma unroll
      for (j = 0; j < NPOS; j++) {    
        atomicAdd(&D[tid + word[j]*nrows], dderiv[j] * cval[j]);
        atomicAdd(&C[tid + cword[j]*nrows], cderiv[j] * dval[j]);
      }
#pragma unroll
      for (k = 0; k < NNEG; k++) {    
        atomicAdd(&D[tid + nword[k]*nrows], ndderiv[k]);
      }
    }   // done with this column
    __syncthreads();
  }
}

#else
template<int NPOS, int NNEG>
__global__ void __word2vecBlock(int nrows, int ncols, int nwords, int shift, int *W, float *D, float *C, float lrate, curandState *rstates) {}

#endif

__global__ void __randinit(curandState *rstates, int nblocks) {
  int id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * blockIdx.x);
  if (id < nblocks) {
    curand_init(1234, id, 0, &rstates[id]); 
  }
}


int word2vecBlock(int nrows, int ncols, int nwords, int shift, int npos, int nneg, int *W, float *D, float *C, float lrate) {
  dim3 threads(32, 10, 1);
  int nblocks = min(1024, nwords / npos);
  curandState *rstates;
  int err = cudaMalloc(( void **)& rstates , nneg * nblocks * sizeof(curandState));
  cudaDeviceSynchronize();
  int ntt = min(128, nneg * nblocks);
  int ntb = 1 + (nneg * nblocks - 1)/ntt;
  __randinit<<<ntb,ntt>>>(rstates, nneg * nblocks); 
  cudaDeviceSynchronize();
  switch (npos * 1000 + nneg) { 
  case 1005:  __word2vecBlock<1,5><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;
  case 1015:  __word2vecBlock<1,15><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;
  case 4015:  __word2vecBlock<4,15><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;
  case 4008:  __word2vecBlock<4,8><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;
  case 8008:  __word2vecBlock<8,8><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;  
  case 8005:  __word2vecBlock<8,5><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;  
  case 4016:  __word2vecBlock<4,16><<<nblocks,threads>>>(nrows, ncols, nwords, shift, W, D, C, lrate, rstates); break;
  } 
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}
*/


#if __CUDA_ARCH__ >= 300
// Custom convolution kernel for vectors

template<int SHIFT>
__global__ void __convRows(int nrows, int ncols, float *A, int lda, float *B, int ldb, float *C) {
  const int window = 2*SHIFT+1; 
  const int height = 32 - 2 * SHIFT;
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
        for (k = -SHIFT; k <= 0; k++) {
          prods[k+SHIFT] += a * __shfl_up(b, -k);
        }
#pragma unroll
        for (k = 1; k <= SHIFT; k++) {
          prods[k+SHIFT] += a * __shfl_down(b, k);
        }
      }
    }
    if (threadIdx.x >= SHIFT && threadIdx.x < 32-SHIFT) {
#pragma unroll
      for (k = 0; k < window; k++) {
        C[k + window * (tid - SHIFT)] = prods[k];
      }
    }
  }
}

template<int SHIFT>
__global__ void __convColsx(int nrows, int ncols, int *W, float *A, float *B, float *C) {
  const int window = 2*SHIFT+1; 
  const int width = 32 - 2*SHIFT;
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
      for (j = threadIdx.z; j < dxy; j += blockDim.z) { // load data into SHMEM
        word = WW[j];
        if (i + tid < nrows) {
          if (j >= SHIFT && j < dxy - SHIFT) {
            AA[j-SHIFT][tid] = A[i + tid + word * nrows];
          }
          BB[j][tid] = B[i + tid + word * nrows];
        }
      }
      __syncthreads();
#pragma unroll
      for (j = 0; j < NTB; j++) {                           // Get some SHMEM data into registers
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

template<int SHIFT>
__global__ void __convCols(int nrows, int ncols, int *W, float *A, float *B, float *C) {
  const int window = 2*SHIFT+1;
  float aa[WLEN + SHIFT];
  float bb[WLEN + SHIFT];
  float prods[WLEN][window];
  int word[WLEN + SHIFT];
  __shared__ float CC[WLEN*BYDIM*window];
  int i, j, k, icol, jcol;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int dxy = blockDim.x * blockDim.y;

  for (icol = 0; icol < ncols; icol += WLEN * gridDim.x) {
    jcol = icol + WLEN * blockIdx.x;
#pragma unroll
    for (j = 0; j < WLEN+SHIFT; j++) {
      if (j + jcol < ncols) {
        word[j] = W[j + jcol];
      } else {
        word[j] = 0;
      }
    } 
#pragma unroll
    for (j = 0; j < WLEN; j++) {
#pragma unroll
      for (k = 0; k <= 2*SHIFT; k++) {
        prods[j][k] = 0;
      }
    }
    for (i = tid; i < nrows; i += dxy) {
#pragma unroll
      for (j = 0; j < WLEN+SHIFT; j++) {
        aa[j] = A[i + word[j] * nrows];
        bb[j] = B[i + word[j] * nrows];
      }
#pragma unroll
      for (j = 0; j < WLEN; j++) {
#pragma unroll
        for (k = 0; k <= SHIFT; k++) {
          prods[j][SHIFT+k] += aa[j] * bb[j+k];
        }
#pragma unroll
        for (k = 1; k <= SHIFT; k++) {
          prods[j][SHIFT-k] += aa[j+k] * bb[j];
        }
      }
    }
#pragma unroll
    for (i = 0; i < WLEN; i++) {
#pragma unroll
      for (j = 0; j <= 2*SHIFT; j++) {
#pragma unroll
        for (k = 1; k < 32; k = k+k) {
          float tmp = __shfl_down(prods[i][j], k);
          prods[i][j] += tmp;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (j = 0; j < WLEN; j++) {
#pragma unroll
        for (k = 0; k < window; k++) {
          CC[k + window * (j + WLEN * threadIdx.y)] = prods[j][k];
        }
      }
    }
    __syncthreads();
    for (j = 0; j < WLEN * window; j += dxy) {
      for (k = 1; k < blockDim.y; k++) {
        if (j + tid < WLEN * window) {
          CC[j + tid] += CC[j + tid + k * WLEN * window];
        }
        __syncthreads();
      } 
      if (j + tid < WLEN * window && j + tid + jcol * window < ncols * window) {
        C[j + tid + jcol * window] = CC[j + tid];
      }
    }
  }
}


#else

template<int SHIFT>
__global__ void __convRows(int nrows, int ncols, float *A, int lda, float *B, int ldb, float *C) {}

template<int SHIFT>
__global__ void __convCols(int nrows, int ncols, int *W, float *A, float *B, float *C) {}

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

int convCols(int nrows, int ncols, int shift, int *W, float *A, float *B, float *C) {
  //  dim3 threads(32, 1, NTZ);
  //  int nblocks = min(2048, 2 + (ncols - 1)/(32 - 2*shift));
  dim3 threads(32, BYDIM, 1);
  int nblocks = min(2048, 2 + (ncols - 1)/5);
  switch(shift) {
  case 5 : __convCols<5><<<nblocks,threads>>>(nrows, ncols, W, A, B, C); break;
    //  case 10 : __convCols<7><<<nblocks,threads>>>(nrows, ncols, W, A, B, C); break;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}
