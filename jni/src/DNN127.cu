#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

#define BYDIMF 2
#define BYDIMB 5
#define CDIM 2

#if __CUDA_ARCH__ >= 300

/*
 * Combined forward-backward word2vec kernel
 */

template<int NWA, int NWB, int MAXDIM>
  __global__ void __word2vec(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float lrate) {
  const int nwab = NWA*NWB;
  __shared__ float CC[NWA*NWB*BYDIMF];
  float aa;
  float bb[NWB];
  float dd[MAXDIM];
  float prods[NWA][NWB];
  float v, sum;
  int wa[NWA];
  int wb[NWB];
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int dxy = blockDim.x * blockDim.y;
  int i, j, k, icol;
  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

  for (icol = istart; icol < iend; icol++) {                // Iterate over columns
#pragma unroll
    for (i = 0; i < NWA; i++) {
      wa[i] = WA[i + icol * NWA];                           // Fill the A word matrix
#pragma unroll
      for (j = 0; j < NWB; j++) {                           // clear the products matrix
        prods[i][j] = 0;
      }
    }
#pragma unroll
    for (i = 0; i < NWB; i++) {
      wb[i] = WB[i + icol * NWB];                           // Fill the B word matrix
    }

    for (i = tid; i < nrows; i += dxy) {                    // Now iterate over the rows of this block
#pragma unroll
      for (j = 0; j < NWB ; j++) {                          // Read B
        bb[j] = B[i + wb[j] * nrows];
      }
#pragma unroll
      for (j = 0; j < NWA; j++) {                           // Compute the products of these elements
        aa = A[i + wa[j] * nrows];
#pragma unroll
        for (k = 0; k < NWB; k++) {
          prods[j][k] += aa * bb[k];
        }
      }
    }                                                       // Finished the entire block

#pragma unroll
    for (i = 0; i < NWA; i++) {                             // Reduce the products within each warp
#pragma unroll
      for (j = 0; j < NWB; j++) {
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
      for (i = 0; i < NWA; i++) {
#pragma unroll
        for (j = 0; j < NWB; j++) {
          CC[j + NWB * (i + NWA * threadIdx.y)] = prods[i][j];
        }
      }
    }
    __syncthreads();
    for (i = 1; i < blockDim.y; i++) {
      __syncthreads();
      for (j = tid; j < nwab; j += dxy) {                   // Reduce the products across warps
        CC[j] += CC[j + i * nwab];
      } 
    } 
    __syncthreads();

    for (i = tid; i < NWA*NWB; i+= dxy) {                   // Compute logistic function on all products
      v = CC[i];
      if (v > 16.0f) {
        v = 1.0f;
      } else {
        v = exp(v);
        v = v / (1.0f + v);
      }
      CC[i] = -v;                                           // All these pairs have label 0
    }

    __syncthreads();
    for (i = tid; i < nrows; i += dxy) {
#pragma unroll
      for (j = 0; j < NWB; j++) {                           // Load B data
        dd[j] = B[i + wb[j] * nrows];
      }
#pragma unroll
      for (j = 0; j < NWA; j++) {                           // Now do the product
        sum = 0;
#pragma unroll
        for (k = 0; k < NWB; k++) {                       
          float xx = CC[j + k * NWA];
          sum += xx * dd[k];
        }
        atomicAdd(&A[i + wa[j] * nrows], sum * lrate);
      }

#pragma unroll
      for (j = 0; j < NWA; j++) {                           // Load A data
        dd[j] = A[i + wa[j] * nrows];
      }
#pragma unroll
      for (j = 0; j < NWB; j++) {                           // Now do the product
        sum = 0;
#pragma unroll
        for (k = 0; k < NWA; k++) {                       
          float xx = CC[k + j * NWA];
          sum += xx * dd[k];
        }
        atomicAdd(&B[i + wb[j] * nrows], sum * lrate);
      }
    } 
    __syncthreads();

  }
}


/*
 *
 * Simple forward kernel for word2vec. Computes inner products of columns from A with columns from B. 
 * The column indices are specified by two "word" matrices. The inner products are computed as an outer product
 * of the word matrices.
 * 
 *  NWA is the number of words per column in WA
 *  NWB is the number of words per column in WB
 *
 *  Columns of the output matrix C are <window> = NWA*NWB long, and contain inner products with corresponding columns of B. 
 *
 */

template<int NWA, int NWB, int BDIM>
__global__ void __word2vecFwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C) {
  const int nwab = NWA*NWB;
  __shared__ float CC[NWA*NWB*BDIM];
  float aa;
  float bb[NWB];
  float prods[NWA][NWB];
  int wa[NWA];
  int wb[NWB];
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int dxy = blockDim.x * blockDim.y;
  int i, j, k, icol;
  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

  for (icol = istart; icol < iend; icol++) {                // Iterate over columns
#pragma unroll
    for (i = 0; i < NWA; i++) {
      wa[i] = WA[i + icol * NWA];                           // Fill the A word matrix
#pragma unroll
      for (j = 0; j < NWB; j++) {                           // clear the products matrix
        prods[i][j] = 0;
      }
    }
#pragma unroll
    for (i = 0; i < NWB; i++) {
      wb[i] = WB[i + icol * NWB];                           // Fill the B word matrix
    }

    for (i = tid; i < nrows; i += dxy) {                    // Now iterate over the rows of this block
#pragma unroll
      for (j = 0; j < NWB ; j++) {                          // Read B
        bb[j] = B[i + wb[j] * nrows];
      }
#pragma unroll
      for (j = 0; j < NWA; j++) {                           // Computes the products of these elements
        aa = A[i + wa[j] * nrows];
#pragma unroll
        for (k = 0; k < NWB; k++) {
          prods[j][k] += aa * bb[k];
        }
      }
    }                                                       // Finished the entire block

#pragma unroll
    for (i = 0; i < NWA; i++) {                             // Reduce the products within each warp
#pragma unroll
      for (j = 0; j < NWB; j++) {
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
      for (i = 0; i < NWA; i++) {
#pragma unroll
        for (j = 0; j < NWB; j++) {
          CC[j + NWB * (i + NWA * threadIdx.y)] = prods[i][j];
        }
      }
    }

    __syncthreads();
    for (i = 1; i < blockDim.y; i++) {
      __syncthreads();
#pragma unroll
      for (j = tid; j < nwab; j += dxy) {                   // Reduce the products across warps
        CC[j] += CC[j + i * nwab];
      } 
    } 
    __syncthreads();
    for (i = tid; i < nwab; i += dxy) {                     // Save to main memory
      C[i + icol * nwab] = CC[i];  
        //atomicAdd(&C[i + icol * nwab], CC[i]); 
    }
    __syncthreads();
  }
}

/*
 *
 * Simple backward kernel for word2vec. 
 * Computes the gradient for A given B or vice-versa, and does an SGD update.
 * 
 *  NWA is the number of words per column in WA
 *  NWB is the number of words per column in WB
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

  for (icol = istart; icol < iend; icol++) {                // iterate in columns
#pragma unroll
    for (j = 0; j < NWA; j++) {
      wa[j] = WA[j + icol * NWA];                           // Load the A word matrix
    }
    __syncthreads();
#pragma unroll 
    for (j = 0; j < NWB; j++) {
      wb[j] = WB[j + icol * NWB];                           // Load the B word matrix
    }
    for (i = fid; i < nwab; i += dxy) {
      cc[i] = C[i + icol * nwab];
    }
    __syncthreads();
    for (i = tid; i < nrows; i += dxy) {
#pragma unroll
      for (j = 0; j < NWB; j++) {                           // Load the data
        dd[j] = B[i + wb[j] * nrows];
      }

#pragma unroll
      for (j = 0; j < NWA; j++) {                           // Now do the product
        sum = 0;
#pragma unroll
        for (k = 0; k < NWB; k++) {                       
          float xx =  cc[j + k * NWA];
          sum += xx * dd[k];
        }
        atomicAdd(&A[i + wa[j] * nrows], sum * lrate);
      }

#pragma unroll
      for (j = 0; j < NWA; j++) {                           // Load the data
        dd[j] = A[i + wa[j] * nrows];
      }

#pragma unroll
      for (j = 0; j < NWB; j++) {                           // Now do the product
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
 * Convolutional kernel for word2vec. This handles the positively-label word pairs with
 * one context word and the current word. 
 */

template<int SKIP, int YDIM, int NREPS>
__global__ void __word2vecConv(int nrows, int ncols, int *W, float *A, float *B, float lrate) {
  const int nwindow = 2*SKIP+1; 
  int words[nwindow];
  float adata[NREPS][nwindow];
  float bdata[NREPS];
  float dbdata[NREPS];
  __shared__ float CC[YDIM * nwindow];

  int i, j, k, tid, indx, icol, dxy;
  float prod, v, av;
  tid = threadIdx.x + blockDim.x * threadIdx.y;
  dxy = blockDim.x * blockDim.y;
  bool good;

  int istart = (int)((1L * blockIdx.x * ncols) / gridDim.x);
  int iend = (int)((1L * (blockIdx.x+1) * ncols) / gridDim.x);

#pragma unroll
  for (i = 0; i < nwindow; i++) {                           // Prefill the word and adata window buffers
    if (istart + i - SKIP - 1 > 0) {
      words[i] = W[istart + i - SKIP - 1];                  // Get a new word
    } else {
      words[i] = -1;
    }
    good = (words[i] >= 0);
#pragma unroll
    for (j = 0; j < NREPS; j++) {                           // Get the A vector for this word
      indx = tid + j * dxy;
      if (good && indx < nrows) {
        adata[j][i] = A[indx + words[i] * nrows];
      } else {
        adata[j][i] = 0;
      }
    }
  }

  for (icol = istart; icol < iend; icol++) {                // Iterate over columns
#pragma unroll
    for (i = 0; i < nwindow-1; i++) {                       // slide words down
      words[i] = words[i+1];
#pragma unroll
      for (j = 0; j < NREPS; j++) {
        adata[j][i] = adata[j][i+1];                         // slide data down
      }
    }

    good = (icol + SKIP < ncols);
    if (good) {
      words[nwindow - 1] = W[icol + SKIP];                  // Get a new word
    } else {
      words[nwindow - 1] = -1;
    }
    good = good && words[nwindow-1] >= 0;

#pragma unroll
    for (j = 0; j < NREPS; j++) {                           // Get a new A column
      indx = tid + j * dxy;
      if (good && indx < nrows) {
        adata[j][nwindow - 1] = A[tid + j * dxy + words[nwindow - 1] * nrows];
      } else {
        adata[j][nwindow - 1] = 0;
      }
      if (words[SKIP] >= 0 && indx < nrows) {               // Get a new B column
        bdata[j] = B[indx + words[SKIP] * nrows];
      } else {
        bdata[j] = 0;
      }
    }
    __syncthreads();

#pragma unroll                 
    for (i = 0; i < nwindow; i++) {                         // Iterate across the window for A cols
      prod = 0;
#pragma unroll                 
      for (j = 0; j < NREPS; j++) {                         // Iterate over blocks of elements
        prod += adata[i][j] * bdata[j];                     // Compute the product between current A, B cols
      }
#pragma unroll                 
      for (k = 1; k < 32; k = k + k) {
        prod += __shfl_down(prod, k);                       // Reduce within warp
      }  
      if (threadIdx.x == 0) {
        CC[i + threadIdx.y * nwindow] = prod;               // Save to SHMEM
      }
    }

    __syncthreads();
    for (i = 0; i < blockDim.y; i++) {                      // Reduce across warps
      for (k = tid; k < nwindow; k += dxy) { 
        CC[k] = CC[k + i * nwindow];
      }
      __syncthreads();
    }

    __syncthreads();                                        //  Apply the sigmoid map
    for (i = tid; i < nwindow; i += dxy) { 
      v = CC[i];
      if (v > 20.0f) {
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
      dbdata[j] = 0;
    }
#pragma unroll                 
    for (i = 0; i < nwindow; i++) {                         // Iterate across the window for A cols
      v = lrate * CC[i];
#pragma unroll                 
      for (j = 0; j < NREPS; j++) {
        av = adata[i][j];
        adata[i][j] += v * bdata[j];                        // Compute the product with the current A, B cols
        dbdata[j] += v * av;
      }
    }
#pragma unroll                 
    for (j = 0; j < NREPS; j++) {
      if (words[SKIP] >= 0 && tid + j * dxy < nrows) {      // Save the B column
        B[tid + j * dxy + words[SKIP] * nrows] = bdata[j] + dbdata[j];
      }
    }
    __syncthreads();  
    if (icol - SKIP >= 0 && words[0] >= 0) {
      for (j = 0; j < NREPS; j++) {                         // Save the A column
        if (tid + j * dxy < nrows) {
          A[tid + j * dxy + words[0] * nrows] = adata[j][0];
        }
      } 
    }
  }
}


#else

template<int NWA, int NWB, int MAXDIM>
  __global__ void __word2vec(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float lrate) {}

template<int NWA, int NWB, int BDIM>
__global__ void __word2vecFwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C) {}

template<int NWA, int NWB, int MAXDIM>
  __global__ void __word2vecBwd(int nrows, int ncols, int *WA, int *WB, float *A, float *B, float *C, float lrate) {}

template<int SKIP, int YDIM, int NREPS>
__global__ void __word2vecConv(int nrows, int ncols, int *W, float *A, float *B, float lrate) {}

#endif

int word2vecConv(int nrows, int ncols, int skip, int *W, float *A, float *B, float lrate) {
  dim3 threads(32, CDIM, 1);
  int nblocks = 1 + (nrows - 1)/threads.y;
  switch(skip) {
  case 5 : __word2vecConv<5, CDIM, 10/CDIM><<<nblocks,threads>>>(nrows, ncols, W, A, B, lrate); break;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}

int word2vecFwd(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float *C) {
  dim3 threads(32, BYDIMF, 1);
  int nblocks = min(4096, 2 + (ncols - 1));
  int which = nwa*10000 + nwb;
  switch (which) {
  case 50001: __word2vecFwd<5,1,BYDIMF><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C); break;
  case 110001: __word2vecFwd<11,1,BYDIMF><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C); break;
  case 50005: __word2vecFwd<5,5,BYDIMF><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C); break;
  case 110005: __word2vecFwd<11,5,BYDIMF><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C); break;
  case 80006: __word2vecFwd<8,6,BYDIMF><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C); break;
  default : printf("word2vecFwd unsupport size combination %d %d\n", nwa, nwb); return 1;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
  }

int word2vecBwd(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float *C, float lrate) {
  dim3 threads(32*BYDIMB, 1, 1);
  int nblocks = min(2048, 2 + (ncols - 1));
  int which = nwa*10000 + nwb;
  switch (which) {
  case 50001: __word2vecBwd<5,1,5><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 110001: __word2vecBwd<11,1,11><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 50005: __word2vecBwd<5,5,5><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 110005: __word2vecBwd<11,5,11><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  case 80006: __word2vecBwd<8,6,8><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, C, lrate); break;
  default : printf("word2vecBwd unsupport size combination %d %d\n", nwa, nwb); return 1;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}

int word2vec(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float lrate) {
  dim3 threads(32, BYDIMF, 1);
  int nblocks = min(2048, 2 + (ncols - 1));
  int which = nwa*10000 + nwb;
  switch (which) {
  case 50001: __word2vec<5,1,5><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, lrate); break;
  case 110001: __word2vec<11,1,11><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, lrate); break;
  case 50005: __word2vec<5,5,5><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, lrate); break;
  case 110005: __word2vec<11,5,11><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, lrate); break;
  case 80006: __word2vec<8,6,8><<<nblocks,threads>>>(nrows, ncols, WA, WB, A, B, lrate); break;
  default : printf("word2vec unsupport size combination %d %d\n", nwa, nwb); return 1;
  }
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}
