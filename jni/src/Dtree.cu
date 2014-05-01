#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

  static const unsigned int c1 = 0xcc9e2d51;
  static const unsigned int c2 = 0x1b873593;
  static const unsigned int r1 = 15;
  static const unsigned int r2 = 13;
  static const unsigned int m = 5;
  static const unsigned int n = 0xe6546b64;

__device__ inline unsigned int h1(unsigned int k, unsigned int hash) {

  k *= c1;
  k = (k << r1) | (k >> (32-r1));
  k *= c2;
 
  hash ^= k;
  hash = ((hash << r2) | (hash >> (32-r2)) * m) + n;
  return hash;
}

__device__ inline unsigned int mmhash(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int mod, unsigned int seed)
{
  unsigned int hash = seed;
 
  hash = h1(v1, hash);
  hash = h1(v2, hash);
  hash = h1(v3, hash);
  
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);
 
  return (hash % mod);
}

#define DBSIZE (8*1024)

__global__ void __treePack(int *idata, int *treenodes, int *icats, int *jc, long long *out, int *fieldlens, 
			   int nrows, int ncols, int ntrees, int nsamps) {
  __shared__ int dbuff[DBSIZE];
  __shared__ int fl[32];
  int j, k, ic, ival;
  int seed = 45123421;

  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  if (tid < 5) {
    fl[tid] = fieldlens[tid];
  }
  __syncthreads();
  int vshift = fl[0];
  int fshift = fl[1] + vshift;
  int nshift = fl[2] + fshift;
  int tshift = fl[3] + nshift;

  int cmask = (1 << fl[0]) - 1;
  int vmask = (1 << fl[1]) - 1;
  int fmask = (1 << fl[2]) - 1;
  int nmask = (1 << fl[3]) - 1;
  int tmask = (1 << fl[4]) - 1;
  
  int nc = (DBSIZE / nrows);
  int itree = threadIdx.y;
  int jfeat = threadIdx.x;

  for (int i = nc * blockIdx.x; i < ncols; i += nc * gridDim.x) {
    int ctodo = min(nc, ncols - i);
    for (j = tid; j < nrows * ctodo; j += blockDim.x*blockDim.y) {
      dbuff[j] = idata[j + i * nrows];
    }
    __syncthreads();
    
    for (j = i; j < i + ctodo; j++) {
      for (itree = threadIdx.y; itree < ntrees; itree += blockDim.y) {
	int inode = treenodes[itree + j * ntrees];
	int ifeat = mmhash(itree, inode, jfeat, nrows, seed);
	long long hdr = (((long long)(tmask & itree)) << tshift) | (((long long)(nmask & inode)) << nshift) | (((long long)(fmask & ifeat)) << fshift);
	for (k = jc[j]; k < jc[j+1]; k++) {    
	  ic = icats[k];
	  if (jfeat < nsamps) {
	    ival = dbuff[ifeat + (j - i) * nrows];
	    out[jfeat + nsamps * (itree + ntrees * k)] = hdr | (((long long)(vmask & ival)) << vshift) | ((long long)(ic & cmask));
	  }
	}
      }
    }
    __syncthreads();
  }
}

int treePack(int *fdata, int *treenodes, int *icats, int *jc, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps) {
  int ntx = 32 * (1 + (nsamps - 1)/32);
  int nty = min(1024 / ntx, ntrees);
  dim3 bdim(ntx, nty, 1);
  int nb = min(32, 1 + (ncols-1)/32);
  __treePack<<<nb,bdim>>>(fdata, treenodes, icats, jc, out, fieldlens, nrows, ncols, ntrees, nsamps);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


class entImpty {
 public:
  static __device__ inline float fupdate(float v) { return v * log((float)v); }
  static __device__ inline float ffinal(float vacc, float vsum) { return log(vsum) - vacc / vsum; }
};

class giniImpty {
 public:
  static __device__ inline float fupdate(float v) { return v * v; }
  static __device__ inline float ffinal(float vacc, float vsum) { return 1 - vacc / (vsum*vsum); }
};

#if __CUDA_ARCH__ >= 300

template<typename T>
__global__ void __minImpurity(long long *keys, int *counts, int *out, float *outv, int *jc, int *fieldlens, 
                              int ntrees, int nnodes, int ncats, int nsamps) {
  __shared__ int catcnt[DBSIZE];

  int tid = threadIdx.x + blockDim.x * threadIdx.y;

  if (tid < 5) {
    catcnt[tid] = fieldlens[tid];
  }
  __syncthreads();
  int vshift = catcnt[0];

  int cmask = (1 << catcnt[0]) - 1;
  int vmask = (1 << catcnt[1]) - 1;
  __syncthreads();

  int i, j, k, h, jc0, jc1, jtodo;
  long long key;
  int ccnt, ctot, cnew, cnt, ival, icat, lastival, bestival, tmp;
  float update, cacc, impty, minimpty, lastimpty, tmpx;

  for (i = threadIdx.y + blockDim.y * blockIdx.x; i < ntrees*nnodes*nsamps; i += blockDim.y * gridDim.x) {
    // Process a group with fixed itree, inode, and ifeat

    jc0 = jc[i];                                            // The range of indices for this group
    jc1 = jc[i+1];
    
    // Clear the cat counts for this group
    for (j = tid; j < DBSIZE; j += blockDim.x * blockDim.y) {
      catcnt[j] = 0;
    }
    __syncthreads();


    lastival = -1;
    lastimpty = 1e7f;
    minimpty = 1e7f;
    ctot = 0;
    cacc = 0.0f;
    for (j = jc0; j < jc1; j += blockDim.x) {
      if (j + threadIdx.x < jc1) {                         // Read a block of (32) keys and counts
        key = keys[j + threadIdx.x];                       // Each (x) thread handles a different input
        cnt = counts[j + threadIdx.x];
        icat = (int)(key & cmask);                         // Extract the cat id and integer value
        ival = ((int)(key >> vshift)) & vmask;
      }
      jtodo = min(32, jc1 - j);
      for (k = 0; k < jtodo; k++) {                        // Sequentially update counts so that each thread
        if (threadIdx.x == k) {                            // in this warp gets the old and new counts
          ccnt = catcnt[icat + ncats * threadIdx.y];       // save data for item k in thread k
          cnew = ccnt + cnt;
          catcnt[icat + ncats * threadIdx.y] = cnew;
        }
      }
      update = T::fupdate((float)cnew);                    // Compute the impurity update for this input
      if (ccnt > 0) update -= T::fupdate((float)ccnt);
#pragma unroll
      for (h = 1; h < 32; h = h + h) {                     // Form the cumsums of updates and counts
        tmpx = __shfl_up(update, h);
        tmp = __shfl_up(cnt, h);
        if (threadIdx.x >=h) {
          update += tmpx;
          cnt += tmp;
        }        
      }  
      ctot += cnt;                                        // Now update the total c and total ci log ci sums
      cacc += update;
      ctot = max(1, ctot);
      impty = T::ffinal(cacc, (float)ctot);              // And the impurity for this input

      tmp = __shfl_up(ival, 1);
      tmpx = __shfl_up(impty, 1);                         // Need the last impurity and ival in order
      if (threadIdx.x > 0) {                              // to restrict the partition feature to a value boundary
        lastival = tmp;
        lastimpty = tmpx;
      }
      if (ival == lastival) lastimpty = 1e7f;             // Eliminate values which are not at value boundaries
      if (lastimpty < minimpty) {
        minimpty = lastimpty;
        bestival = lastival;
      }

#pragma unroll
      for (h = 1; h < 32; h = h + h) {                    // Find the cumulative min impurity and corresponding ival
        tmpx = __shfl_up(minimpty, h);
        tmp = __shfl_up(bestival, h);
        if (threadIdx.x >= h && tmpx < minimpty) {
          minimpty = tmpx;
          bestival = tmp;
        }        
      }
      minimpty = __shfl(minimpty, jtodo-1);               // Carefully copy the last active thread to all threads, needed outside this loop     
      bestival = __shfl(bestival, jtodo-1);
      ctot = __shfl(ctot, jtodo-1);                
      cacc = __shfl(cacc, jtodo-1);
      lastival = __shfl(ival, jtodo-1);             
      lastimpty = __shfl(impty, jtodo-1);
    }
    if (threadIdx.x == 0) {
      out[i] = bestival;                                  // Output the best split feature value
      outv[i] = minimpty - T::ffinal(cacc, (float)ctot);  // And the impurity gain
    }
  }
}
#else
template<class T>
__global__ void __minImpurity(long long *keys, int *counts, int *out, float *outv, int *jc, int *fieldlens, 
                              int ntrees, int nnodes, int ncats, int nsamps) {}
#endif

int minImpurity(long long *keys, int *counts, int *out, float *outv, int *jc, int *fieldlens, 
                int ntrees, int nnodes, int ncats, int nsamps, int impType) {
  int ny = min(32, DBSIZE/ncats);
  dim3 tdim(32, ny, 1);
  int ng = min(64, 1L*ntrees*nnodes*nsamps);
  if (impType == 0) {
    __minImpurity<entImpty><<<ng,tdim>>>(keys, counts, out, outv, jc, fieldlens, ntrees, nnodes, ncats, nsamps);
  } else {
    __minImpurity<giniImpty><<<ng,tdim>>>(keys, counts, out, outv, jc, fieldlens, ntrees, nnodes, ncats, nsamps);
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}
