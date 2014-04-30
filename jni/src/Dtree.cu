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


