#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define NEG_INFINITY 0xff800000

#if __CUDA_ARCH__ > 200

// Compute one level of random forest evaluation for a set of 32 trees. 
// The input is a dense feature matrix (feats) which is nrows (features) by ncols (samples).
//   ns is the number of random features used for each tree node.
//   tstride is the tree stride, i.e. how far to step in the trees array to access data from the next tree. 
//   ntrees is the number of trees (can be less than 32 but this wont be efficient in that case). 
//   trees is an array containing the feature indices. It is an ntrees x tstride matrix
//   tpos is an ntrees x ncols matrix containing the position indices for the parent nodes in the trees.
//   i.e. tpos indicates where each sample is in the traversal up to this depth in the trees. 
//   otpos is the output, which is an index for one of two child nodes for each parent, based on whether
//   the current feature sample sum is greater than the threshold.

// The trees array is really nnodes x ns x ntrees (i.e. tstride = nnodes x ns), where nnodes is the number of
// nodes in a single tree at the current depth. 
//  
// In each column of ns feature indices for a tree node, the 0^th index is actually the floating point threshold for the node. 
// It is converted and saved in a variable named fthresh

// ATHREADS and BTHREADS match blockDim.x and blockDim.y (they're used for sizing the arrays). 
// REPTREES is the number of trees processed by each "y" thread group.

template<int ATHREADS, int BTHREADS, int REPTREES>
__global__ void __treeprod(int *trees, float *feats, int *tpos, float *otv, int nrows, int ncols, int ns, int tstride, int ntrees) {
  __shared__ int pos[REPTREES][ATHREADS];
  __shared__ float totals[REPTREES][ATHREADS];
  int bd, tind, ttop;
  float ftmp;
  float vv[REPTREES];
  

  for (bd = blockIdx.x; bd < ncols; bd += gridDim.x) {
    // Read in the index of parent for each tree
    if (threadIdx.x + threadIdx.y*ATHREADS < ntrees) {
      pos[threadIdx.y][threadIdx.x] = tpos[threadIdx.x + threadIdx.y*ATHREADS + ntrees * bd];
    }

    // Now read the tree node vectors associated with these trees
    __syncthreads();
#pragma unroll
    for (int k = 0; k < REPTREES; k++) {
      vv[k] = 0;
      if (threadIdx.y + k*BTHREADS < ntrees) {
        for (int j = threadIdx.x; j < ns+1; j += blockDim.x) {
          tind = trees[j + (ns+1)*pos[k][threadIdx.y] + (threadIdx.y+k*BTHREADS)*tstride];
          ttop = __shfl(tind, 0);
          if (ttop == NEG_INFINITY) {
            vv[k] = ttop;
            break;
          }
          if (j > 0) {
            vv[k] += feats[tind + bd * nrows];  
          }              
        }
      }
    }
    // vv[k] is a thread variable, so sum it over the warp threads
#pragma unroll
    for (int k = 0; k < REPTREES; k++) {
      ftmp = vv[k];
      if (*((int *)&ftmp) != NEG_INFINITY) {            // This is a leaf node, dont do anything (leaf marker will be output)
#pragma unroll
        for (int i = 1; i < 32; i *= 2) {
          vv[k] += __shfl_down(vv[k], i);
        }
      }
    }

    if (threadIdx.x == 0) {
#pragma unroll
      for (int k = 0; k < REPTREES; k++) {   // and save in the totals array
        totals[k][threadIdx.y] = vv[k];
      }
    }

    // save
    __syncthreads();
    if (threadIdx.x + threadIdx.y*ATHREADS < ntrees) {
      otv[threadIdx.x + threadIdx.y*ATHREADS + ntrees * bd] = totals[threadIdx.y][threadIdx.x];
    }  
    __syncthreads();
  }
} 


template<int ATHREADS, int BTHREADS, int REPTREES>
__global__ void __treesteps(int *trees, float *feats, int *tpos, int *otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int tdepth) {

  __shared__ int pos[REPTREES][ATHREADS];
  __shared__ float thresh[REPTREES][ATHREADS];
  __shared__ float totals[REPTREES][ATHREADS];
  int newt, bd, tind, ttop, kk;
  float ftmp;
  float vv[REPTREES];

  for (bd = blockIdx.x; bd < ncols; bd += gridDim.x) {
    // Read in the index of parent for each tree
    if (threadIdx.x + threadIdx.y*ATHREADS < ntrees) {
      pos[threadIdx.y][threadIdx.x] = tpos[threadIdx.x + threadIdx.y*ATHREADS + ntrees * bd];
    }
    for (int id = 0; id < tdepth; id ++) {
      // Now read the tree node vectors associated with these trees
      __syncthreads();
#pragma unroll
      for (int k = 0; k < REPTREES; k++) {
        vv[k] = 0;
        kk = threadIdx.y + k*BTHREADS; 
        if (kk < ntrees) {
          for (int j = threadIdx.x; j < ns+1; j += blockDim.x) {
            tind = trees[j + (ns+1)*pos[k][threadIdx.y] + (threadIdx.y+k*BTHREADS)*tstride];
            if (j == 0) {
              thresh[k][threadIdx.y] = *((float *)&tind); // Save the node threshold
            }
            ttop = __shfl(tind, 0);                         
            if (ttop == NEG_INFINITY) {            // This is a leaf
              if (j == 1) {
                pos[k][threadIdx.y] = tind;        // Save the class label
              }
              break;
            }
            if (j > 0) {
              vv[k] += feats[tind + bd * nrows];  // Non-leaf, compute the node score
            }              
          }
        }
      }

      // Since vv[k] is a thread variable, sum it over threads
#pragma unroll
      for (int k = 0; k < REPTREES; k++) {
#pragma unroll
        for (int i = 1; i < 32; i *= 2) {
          vv[k] += __shfl_down(vv[k], i);
        }
      }
      if (threadIdx.x == 0) {
#pragma unroll
        for (int k = 0; k < REPTREES; k++) {   // and save in the totals array
          totals[k][threadIdx.y] = vv[k];
        }
      }

      // check thresholds and save as needed
      __syncthreads();
      if (threadIdx.x + threadIdx.y*ATHREADS < ntrees) {
        ftmp = thresh[threadIdx.y][threadIdx.x];
        if (*((int *)&ftmp) != NEG_INFINITY) {  // Check if non-leaf
          newt = 2 * pos[threadIdx.y][threadIdx.x] + 1;
          if (totals[threadIdx.y][threadIdx.x] > thresh[threadIdx.y][threadIdx.x]) {
            newt++;
          }
          pos[threadIdx.y][threadIdx.x] = newt; 
        }                          // Do nothing if its a leaf, pos already contains the class label
      }  
      __syncthreads();
    }
    if (threadIdx.x + threadIdx.y*ATHREADS < ntrees) {
      otpos[threadIdx.x + threadIdx.y*ATHREADS + ntrees * bd] = pos[threadIdx.y][threadIdx.x];
    }
  }
} 

#else
template<int ATHREADS, int BTHREADS, int REPTREES>
__global__ void __treeprod(int *trees, float *feats, int *tpos, float *otval, int nrows, int ncols, int ns, int tstride, int ntrees) {}
template<int ATHREADS, int BTHREADS, int REPTREES>
__global__ void __treesteps(int *trees, float *feats, int *tpos, int *otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int tdepth) {}
#endif

int treeprod(int *trees, float *feats, int *tpos, float *otv, int nrows, int ncols, int ns, int tstride, int ntrees) {
  int nblks = min(1024, max(ncols/8, min(32, ncols)));
  dim3 blocks(32, 32, 1);
  int ntt;
  for (ntt = 32; ntt < ntrees; ntt *= 2) {}
  switch (ntt) {
  case (32) :
    __treeprod<32,32,1><<<nblks,blocks>>>(trees, feats, tpos, otv, nrows, ncols, ns, tstride, ntrees); break;
  case (64) :
    __treeprod<32,32,2><<<nblks,blocks>>>(trees, feats, tpos, otv, nrows, ncols, ns, tstride, ntrees); break;
  case (128) :
    __treeprod<32,32,4><<<nblks,blocks>>>(trees, feats, tpos, otv, nrows, ncols, ns, tstride, ntrees); break;
  case (256) :
    __treeprod<32,32,8><<<nblks,blocks>>>(trees, feats, tpos, otv, nrows, ncols, ns, tstride, ntrees); break;
  case (512) :
    __treeprod<32,32,16><<<nblks,blocks>>>(trees, feats, tpos, otv, nrows, ncols, ns, tstride, ntrees); break;
  case (1024) :
    __treeprod<32,32,32><<<nblks,blocks>>>(trees, feats, tpos, otv, nrows, ncols, ns, tstride, ntrees); break;
  } 
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}


int treesteps(int *trees, float *feats, int *tpos, int *otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int tdepth) {
  int nblks = min(1024, max(ncols/8, min(32, ncols)));
  dim3 blocks(32, 32, 1);
  int ntt;
  for (ntt = 32; ntt < ntrees; ntt *= 2) {}
  switch (ntt) {
  case (32) :
    __treesteps<32,32,1><<<nblks,blocks>>>(trees, feats, tpos, otpos, nrows, ncols, ns, tstride, ntrees, tdepth); break;
  case (64) :
    __treesteps<32,32,2><<<nblks,blocks>>>(trees, feats, tpos, otpos, nrows, ncols, ns, tstride, ntrees, tdepth); break;
  case (128) :
    __treesteps<32,32,4><<<nblks,blocks>>>(trees, feats, tpos, otpos, nrows, ncols, ns, tstride, ntrees, tdepth); break;
  case (256) :
    __treesteps<32,32,8><<<nblks,blocks>>>(trees, feats, tpos, otpos, nrows, ncols, ns, tstride, ntrees, tdepth); break;
  case (512) :
    __treesteps<32,32,16><<<nblks,blocks>>>(trees, feats, tpos, otpos, nrows, ncols, ns, tstride, ntrees, tdepth); break;
  case (1024) :
    __treesteps<32,32,32><<<nblks,blocks>>>(trees, feats, tpos, otpos, nrows, ncols, ns, tstride, ntrees, tdepth); break;
  } 
  cudaDeviceSynchronize();
  int err = cudaGetLastError();
  return err;
}

#define BLOCKDIM 32
const int INBLOCK = 4;

// copy and transpose columns of the input matrix into the output matrix. nrows refers to the input matrix 
// (and so is ncols for the output). ncols is the length of the iptrs array, which will be the number of 
// rows of the output matrix. iptrs specifies the columns of the input array to copy. 
// outstride is stride of the output matrix

__global__ void __icopy_transpose(int *iptrs, float *in, float *out, int outstride, int nrows, int ncols) {
  __shared__ float tile[BLOCKDIM][BLOCKDIM+1];
  int nx = BLOCKDIM * gridDim.x;
  int ny = BLOCKDIM * gridDim.y;
  int ix = BLOCKDIM * blockIdx.x;
  int iy = BLOCKDIM * blockIdx.y;

  for (int yb = iy; yb < ncols; yb += ny) {
    for (int xb = ix; xb < nrows; xb += nx) {
      if (xb + threadIdx.x < nrows) {
        int ylim = min(ncols, yb + BLOCKDIM);
        for (int y = threadIdx.y + yb; y < ylim; y += blockDim.y) {
          tile[threadIdx.x][y-yb] = in[threadIdx.x + xb + iptrs[y]*nrows];
        }
      }
      __syncthreads();
      if (yb + threadIdx.x < ncols) {
        int xlim = min(nrows, xb + BLOCKDIM);
        for (int x = threadIdx.y + xb; x < xlim; x += blockDim.y) {
          out[threadIdx.x + yb + x*outstride] = tile[x-xb][threadIdx.x];
        }
      }
      __syncthreads();
    }
  } 
}

int icopy_transpose(int *iptrs, float *in, float *out, int stride, int nrows, int ncols) {
  const dim3 griddims(20,256,1);
  const dim3 blockdims(BLOCKDIM,INBLOCK,1);
  cudaError_t err;
  __icopy_transpose<<<griddims,blockdims>>>(iptrs, in, out, stride, nrows, ncols); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {fprintf(stderr, "cuda error in icopy_transpose"); return err;}
  return 0;
}

// copy and transpose the input matrix into columns of the output matrix. nrows, ncols refer to output matrix

__global__ void __ocopy_transpose(int *optrs, float *in, float *out, int instride, int nrows, int ncols) {
  int nx = BLOCKDIM * gridDim.x;
  int ny = BLOCKDIM * gridDim.y;
  int ix = BLOCKDIM * blockIdx.x;
  int iy = BLOCKDIM * blockIdx.y;
  __shared__ float tile[BLOCKDIM][BLOCKDIM+1];

  for (int yb = iy; yb < ncols; yb += ny) {
    for (int xb = ix; xb < nrows; xb += nx) {
      if (yb + threadIdx.x < ncols) {
        int xlim = min(nrows, xb + BLOCKDIM);
        for (int x = threadIdx.y + xb; x < xlim; x += blockDim.y) {
          tile[x-xb][threadIdx.x] = in[threadIdx.x + yb + x*instride];
        }
      }
      __syncthreads();
      if (xb + threadIdx.x < nrows) {
        int ylim = min(ncols, yb + BLOCKDIM);
        for (int y = threadIdx.y + yb; y < ylim; y += blockDim.y) {
          out[optrs[y]*nrows + threadIdx.x + xb] = tile[threadIdx.x][y-yb];
        }
      }
      __syncthreads();
    }
  } 
}

__global__ void __ocopy_transpose_add(int *optrs, float *in, float *out, int instride, int nrows, int ncols) {
  int nx = BLOCKDIM * gridDim.x;
  int ny = BLOCKDIM * gridDim.y;
  int ix = BLOCKDIM * blockIdx.x;
  int iy = BLOCKDIM * blockIdx.y;
  __shared__ float tile[BLOCKDIM][BLOCKDIM+1];

  for (int yb = iy; yb < ncols; yb += ny) {
    for (int xb = ix; xb < nrows; xb += nx) {
      if (yb + threadIdx.x < ncols) {
        int xlim = min(nrows, xb + BLOCKDIM);
        for (int x = threadIdx.y + xb; x < xlim; x += blockDim.y) {
          tile[x-xb][threadIdx.x] = in[threadIdx.x + yb + x*instride];
        }
      }
      __syncthreads();
      if (xb + threadIdx.x < nrows) {
        int ylim = min(ncols, yb + BLOCKDIM);
        for (int y = threadIdx.y + yb; y < ylim; y += blockDim.y) {
          atomicAdd(&out[optrs[y]*nrows + threadIdx.x + xb], tile[threadIdx.x][y-yb]);
        }
      }
      __syncthreads();
    }
  } 
}

__global__ void __ocopy_transpose_min(int *optrs, float *in, float *out, int instride, int nrows, int ncols) {
  int nx = BLOCKDIM * gridDim.x;
  int ny = BLOCKDIM * gridDim.y;
  int ix = BLOCKDIM * blockIdx.x;
  int iy = BLOCKDIM * blockIdx.y;
  __shared__ float tile[BLOCKDIM][BLOCKDIM+1];

  for (int yb = iy; yb < ncols; yb += ny) {
    for (int xb = ix; xb < nrows; xb += nx) {
      if (yb + threadIdx.x < ncols) {
        int xlim = min(nrows, xb + BLOCKDIM);
        for (int x = threadIdx.y + xb; x < xlim; x += blockDim.y) {
          tile[x-xb][threadIdx.x] = in[threadIdx.x + yb + x*instride];
        }
      }
      __syncthreads();
      if (xb + threadIdx.x < nrows) {
        int ylim = min(ncols, yb + BLOCKDIM);
        for (int y = threadIdx.y + yb; y < ylim; y += blockDim.y) {
          atomicMin((int *)&out[optrs[y]*nrows + threadIdx.x + xb], *(int *)(&tile[threadIdx.x][y-yb]));
        }
      }
      __syncthreads();
    }
  } 
}

int ocopy_transpose_add(int *optrs, float *in, float *out, int stride, int nrows, int ncols) {
  const dim3 griddims(20,256,1);
  const dim3 blockdims(BLOCKDIM,INBLOCK,1);
  cudaError_t err;
  __ocopy_transpose_add<<<griddims,blockdims>>>(optrs, in, out, stride, nrows, ncols); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {fprintf(stderr, "cuda error in ocopy_transpose"); return err;}
  return 0;
}

int ocopy_transpose(int *optrs, float *in, float *out, int stride, int nrows, int ncols) {
  const dim3 griddims(20,256,1);
  const dim3 blockdims(BLOCKDIM,INBLOCK,1);
  cudaError_t err;
  __ocopy_transpose<<<griddims,blockdims>>>(optrs, in, out, stride, nrows, ncols); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {fprintf(stderr, "cuda error in ocopy_transpose"); return err;}
  return 0;
}

int ocopy_transpose_min(int *optrs, float *in, float *out, int stride, int nrows, int ncols) {
  const dim3 griddims(20,256,1);
  const dim3 blockdims(BLOCKDIM,INBLOCK,1);
  cudaError_t err;
  __ocopy_transpose_min<<<griddims,blockdims>>>(optrs, in, out, stride, nrows, ncols); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {fprintf(stderr, "cuda error in ocopy_transpose"); return err;}
  return 0;
}

