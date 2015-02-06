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

__forceinline__ __device__ unsigned int h1(unsigned int k, unsigned int hash) {

  k *= c1;
  k = (k << r1) | (k >> (32-r1));
  k *= c2;
 
  hash ^= k;
  hash = ((hash << r2) | (hash >> (32-r2)) * m) + n;
  return hash;
}

__forceinline__ __device__ unsigned int mmhash3(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int mod, unsigned int seed)
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

__global__ void __treePack(float *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, 
                           int nrows, int ncols, int ntrees, int nsamps, int seed) {
  __shared__ float fbuff[DBSIZE];
  __shared__ int fl[32];
  int i, j, ic;

  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  if (tid < 6) {
    fl[tid] = fieldlens[tid];
  }
  __syncthreads();
  int vshift = fl[5];
  int ishift = fl[4] + vshift;
  int jshift = fl[3] + ishift;
  int nshift = fl[2] + jshift;
  int tshift = fl[1] + nshift;

  int cmask = (1 << fl[5]) - 1;
  int vmask = (1 << fl[4]) - 1;
  int imask = (1 << fl[3]) - 1;
  int jmask = (1 << fl[2]) - 1;
  int nmask = (1 << fl[1]) - 1;
  int tmask = (1 << fl[0]) - 1;
  
  int nc = (DBSIZE / nrows);
  int itree = threadIdx.y;
  int jfeat = threadIdx.x;

  const int signbit = 0x80000000;
  const int mag =     0x7fffffff;
  int fshift = 32 - fl[4];

  for (i = nc * blockIdx.x; i < ncols; i += nc * gridDim.x) {
    int ctodo = min(nc, ncols - i);
    for (j = tid; j < nrows * ctodo; j += blockDim.x*blockDim.y) {
      fbuff[j] = fdata[j + i * nrows];
    }
    __syncthreads();
    
    for (j = i; j < i + ctodo; j++) {                               // j is the column index
      ic = icats[j];
      for (itree = threadIdx.y; itree < ntrees; itree += blockDim.y) {
        if (jfeat < nsamps) {
          int inode = treenodes[itree + j * ntrees];
	  int ifeat = mmhash3(itree, inode, jfeat, nrows, seed);
          float v = fbuff[ifeat + (j - i) * nrows];
          int vi = *((int *)&v);
          if (vi & signbit) {
            vi = -(vi & mag);
          }
          vi += signbit;
          int ival = vi >> fshift;
          //          int ival = (int)v;
          long long hdr = 
            (((long long)(tmask & itree)) << tshift) | (((long long)(nmask & inode)) << nshift) | 
            (((long long)(jmask & jfeat)) << jshift) | (((long long)(imask & ifeat)) << ishift) |
            (((long long)(vmask & ival)) << vshift) | ((long long)(ic & cmask));
          out[jfeat + nsamps * (itree + ntrees * j)] = hdr;
        }
      }
    }
    __syncthreads();
  }
}


__global__ void __treePackInt(int *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, 
                              int nrows, int ncols, int ntrees, int nsamps, int seed) {
  __shared__ int fbuff[DBSIZE];
  __shared__ int fl[32];
  int i, j, ic, ival;

  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  if (tid < 6) {
    fl[tid] = fieldlens[tid];
  }
  __syncthreads();
  int vshift = fl[5];
  int ishift = fl[4] + vshift;
  int jshift = fl[3] + ishift;
  int nshift = fl[2] + jshift;
  int tshift = fl[1] + nshift;

  int cmask = (1 << fl[5]) - 1;
  int vmask = (1 << fl[4]) - 1;
  int imask = (1 << fl[3]) - 1;
  int jmask = (1 << fl[2]) - 1;
  int nmask = (1 << fl[1]) - 1;
  int tmask = (1 << fl[0]) - 1;
  
  int nc = (DBSIZE / nrows);
  int itree = threadIdx.y;
  int jfeat = threadIdx.x;

  for (i = nc * blockIdx.x; i < ncols; i += nc * gridDim.x) {
    int ctodo = min(nc, ncols - i);
    for (j = tid; j < nrows * ctodo; j += blockDim.x*blockDim.y) {
      fbuff[j] = fdata[j + i * nrows];
    }
    __syncthreads();
    
    for (j = i; j < i + ctodo; j++) {                               // j is the column index
      ic = icats[j];
      for (itree = threadIdx.y; itree < ntrees; itree += blockDim.y) {
        if (jfeat < nsamps) {
          int inode = treenodes[itree + j * ntrees];
          int ifeat = mmhash3(itree, inode, jfeat, nrows, seed);
          ival = fbuff[ifeat + (j - i) * nrows];
          long long hdr = 
            (((long long)(tmask & itree)) << tshift) | (((long long)(nmask & inode)) << nshift) | 
            (((long long)(jmask & jfeat)) << jshift) | (((long long)(imask & ifeat)) << ishift) |
            (((long long)(vmask & ival)) << vshift) | ((long long)(ic & cmask));
          out[jfeat + nsamps * (itree + ntrees * j)] = hdr;
        }
      }
    }
    __syncthreads();
  }
}

int treePack(float *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed) {
  int ntx = 32 * (1 + (nsamps - 1)/32);
  int nty = min(1024 / ntx, ntrees);
  dim3 bdim(ntx, nty, 1);
  int nb = min(32, 1 + (ncols-1)/32);
  __treePack<<<nb,bdim>>>(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


int treePackInt(int *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed) {
  int ntx = 32 * (1 + (nsamps - 1)/32);
  int nty = min(1024 / ntx, ntrees);
  dim3 bdim(ntx, nty, 1);
  int nb = min(32, 1 + (ncols-1)/32);
  __treePackInt<<<nb,bdim>>>(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

#if __CUDA_ARCH__ > 200

__global__ void __treeWalk(float *fdata, int *inodes, float *fnodes, int *itrees, int *ftrees, int *vtrees, float *ctrees,
                           int nrows, int ncols, int ntrees, int nnodes, int getcat, int nbits, int nlevels) {
  __shared__ float fbuff[DBSIZE];
  int i, j, k, base, m, ipos, itree, ftree, vtree, feat, ikid, big;
  float ctree;

  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int inode = threadIdx.x;
  int itr = threadIdx.y;
  int icol = threadIdx.z;
  
  int nc = (DBSIZE / nrows);

  const int signbit = 0x80000000;
  const int mag =     0x7fffffff;
  int fshift = 32 - nbits;
  int mask = (1 << nbits) - 1;

  for (i = nc * blockIdx.x; i < ncols; i += nc * gridDim.x) {
    int ctodo = min(nc, ncols - i);
    // Fill up the SHMEM buffer
    for (j = tid; j < nrows * ctodo; j += blockDim.x*blockDim.y*blockDim.z) {
      fbuff[j] = fdata[j + i * nrows];
    }
    __syncthreads();
    
    for (j = icol; j < i + ctodo; j += blockDim.z) {           // j is the column index
      base = 0;                                                // points to the current B-tree
      for (k = 0; k < nlevels; k++) {
        ipos = base + itr * nnodes + inode;                    // read in a 32-element B-tree block
        itree = itrees[ipos];
        ftree = ftrees[ipos];
        vtree = vtrees[ipos];
        feat = *((int *)&fbuff[ftree + j * nrows]);            // get the feature pointed to 
        if (feat & signbit) {                                  // convert to fixed point
          feat = -(feat & mag);
        }
        feat += signbit;
        feat = (feat >> fshift) & mask;
        big = feat > vtree;                                    // compare with the threshold
        ikid = itree;                                          // address of left child in the block
        // walk down the tree - by passing up the appropriate child
        if (!getcat || k < nlevels-1) {
#pragma unroll
          for (m = 0; m < 5; m++) {
            itree = __shfl(itree, ikid + big);
          }
          base = itree;
        } else {
          ctree = ctrees[ipos];
#pragma unroll
          for (m = 0; m < 5; m++) {
            ctree = __shfl(ctree, ikid + big);
          }
        }
      }
      if (inode == 0) {                                         // save the leaf node index or the label
        if (getcat) {
          fnodes[itr + (i + icol) * ntrees] = ctree;
        } else {
          inodes[itr + (i + icol) * ntrees] = base;
        }
      }
    }
    __syncthreads();
  }
}

#endif

class entImpty {
 public:
  static __device__ inline float fupdate(int v) { return (float)v * logf((float)max(1, v)); }
  static __device__ inline float fresult(float vacc, int vsum) { float vs = (float)max(1, vsum); return logf(vs) - vacc / vs; }
};

class giniImpty {
 public:
  static __device__ inline float fupdate(int v) { return (float)v * (float)v; }
  static __device__ inline float fresult(float vacc, int vsum) { float vs = (float)max(1, vsum); return 1.0f - vacc / (vs*vs); }
};

#if __CUDA_ARCH__ >= 300

__device__ inline void accumup2(int &cnt, float &update) {
#pragma unroll
  for (int h = 1; h < 32; h = h + h) {
    float tmpx = __shfl_up(update, h);
    int tmp = __shfl_up(cnt, h);
    if (threadIdx.x >=h) {
      update += tmpx;
      cnt += tmp;
    }
  }
}

__device__ inline void accumup3(int &cnt, float &update, float &updatet) {
#pragma unroll
  for (int h = 1; h < 32; h = h + h) {
    float tmpx = __shfl_up(update, h);
    float tmpy = __shfl_up(updatet, h);
    int tmp = __shfl_up(cnt, h);
    if (threadIdx.x >=h) {
      update += tmpx;
      updatet += tmpy;
      cnt += tmp;
    }
  }
}

__device__ inline void accumdown3(int &cnt, float &update, float &updatet, int bound) {
#pragma unroll
  for (int h = 1; h < 32; h = h + h) {
    float tmpx = __shfl_down(update, h);
    float tmpy = __shfl_down(updatet, h);
    int tmp = __shfl_down(cnt, h);
    if (threadIdx.x + h <= bound) {
      update += tmpx;
      updatet += tmpy;
      cnt += tmp;
    }
  }
}

__device__ inline void minup2(float &impty, int &ival) {
#pragma unroll
  for (int h = 1; h < 32; h = h + h) {
    float tmpx = __shfl_up(impty, h);
    int tmp = __shfl_up(ival, h);
    if (threadIdx.x >= h && tmpx < impty) {
      impty = tmpx;
      ival = tmp; 
    }
  }
}

__device__ inline void maxup2(int &v, int &indx) {
#pragma unroll
  for (int h = 1; h < 32; h = h + h) {
    int tmpv = __shfl_up(v, h);
    int tmpi = __shfl_up(indx, h);
    if (threadIdx.x >= h && tmpv > v) {
      v = tmpv;
      indx = tmpi; 
    }
  }
}

template<typename T>
__global__ void __minImpuritya(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens,
                              int nnodes, int ncats, int nsamps) {
  __shared__ int catcnt[DBSIZE/2];
  __shared__ int cattot[DBSIZE/2];
  int tid = threadIdx.x + blockDim.x * threadIdx.y;

  if (tid < 6) {
    catcnt[tid] = fieldlens[tid];
  }
  __syncthreads();
  int vshift = catcnt[5];
  int ishift = catcnt[4] + vshift;

  int cmask = (1 << catcnt[5]) - 1;
  int vmask = (1 << catcnt[4]) - 1;
  int imask = (1 << catcnt[3]) - 1;

  __syncthreads();

  int i, j, k, jc0, jc1, jlast;
  long long key;
  int cold, ctot, ctt, ctotall, cnew, cnt, ival, icat, lastival, bestival, tmp, maxcnt, imaxcnt;
  float update, updatet, cacc, cact, caccall, impty, minimpty, lastimpty, tmpx;

  for (i = threadIdx.y + blockDim.y * blockIdx.x; i < nnodes*nsamps; i += blockDim.y * gridDim.x) {
    // Process a group with fixed itree, inode, and ifeat

    jc0 = jc[i]; // The range of indices for this group
    jc1 = jc[i+1];
    __syncthreads();

    // Clear the cat counts for this group
    for (j = tid; j < DBSIZE/2; j += blockDim.x * blockDim.y) {
      catcnt[j] = 0;
      cattot[j] = 0;
    }
    __syncthreads();

    // First pass gets counts for each category and the (ci)log(ci) sum for this block
    ctot = 0; 
    cacc = 0.0f; 
    maxcnt = -1;
    imaxcnt = -1;
    for (j = jc0; j < jc1; j += blockDim.x) {
      if (j + threadIdx.x < jc1) {                          // Read a block of (32) keys and counts
        key = keys[j + threadIdx.x];                        // Each (x) thread handles a different input
        cnt = counts[j + threadIdx.x];
        icat = ((int)key) & cmask;                          // Extract the cat id and integer value
      }
      jlast = min(31, jc1 - j - 1);
      for (k = 0; k <= jlast; k++) {                        // Sequentially update counts so that each thread
        if (threadIdx.x == k) {                             // in this warp gets the old and new counts
          cold = cattot[icat + ncats * threadIdx.y];        // i.e. data for item k is in thread k
          cnew = cold + cnt;
          cattot[icat + ncats * threadIdx.y] = cnew;
        }
      }
      update = T::fupdate(cnew) - T::fupdate(cold);
      accumup2(cnt,update);
      ctot += cnt;                                          // Now update the total c and total ci log ci sums
      cacc += update;
      ctot = __shfl(ctot, jlast);
      cacc = __shfl(cacc, jlast);
      if (cnew > maxcnt) {                                  // Compute and distribute the max cnt
        maxcnt = cnew;
        imaxcnt = icat;
      }
      maxup2(maxcnt, imaxcnt);
      maxcnt = __shfl(maxcnt, jlast);
      imaxcnt = __shfl(imaxcnt, jlast);
    }
    __syncthreads();
    //    if (threadIdx.x == 0 && i < 32) printf("cuda %d %d %f\n", i, ctot, cacc);

    // Second pass to compute impurity at every input point
    caccall = cacc;                                         // Save the total count and (ci)log(ci) sum
    cact = cacc;
    ctotall = ctot;
    ctot = 0;
    cacc = 0.0f;
    lastival = -1;
    lastimpty = 1e7f;
    minimpty = 1e7f;
    for (j = jc0; j < jc1; j += blockDim.x) {
      if (j + threadIdx.x < jc1) {                          // Read a block of (32) keys and counts
        key = keys[j + threadIdx.x];                        // Each (x) thread handles a different input
        cnt = counts[j + threadIdx.x];
        icat = ((int)key) & cmask;                          // Extract the cat id and integer value
        ival = ((int)(key >> vshift)) & vmask;
      }
      jlast = min(31, jc1 - j - 1);
      for (k = 0; k <= jlast; k++) {                        // Sequentially update counts so that each thread
        if (threadIdx.x == k) {                             // in this warp gets the old and new counts
          cold = catcnt[icat + ncats * threadIdx.y];        // i.e. data for item k is in thread k
          ctt = cattot[icat + ncats * threadIdx.y];
          cnew = cold + cnt;
          catcnt[icat + ncats * threadIdx.y] = cnew;
        }
      }
      update = T::fupdate(cnew) - T::fupdate(cold);         // Compute the impurity updates for this input
      updatet = T::fupdate(ctt-cnew) - T::fupdate(ctt-cold);

      accumup3(cnt, update, updatet);
      ctot += cnt;                                          // Now update the total c and total ci log ci sums
      cacc += update;
      cact += updatet;
      impty = T::fresult(cacc, ctot) + T::fresult(cact, ctotall-ctot); // And the impurity for this input
      //      if (i == 0) printf("cuda pos %d impty %f icat %d cnts %d %d cacc %f %d\n", j + threadIdx.x, impty, icat, cold, cnew, cacc, ctot);

      tmp = __shfl_up(ival, 1);                             // Need the last impurity and ival in order
      tmpx = __shfl_up(impty, 1);                           // to restrict the partition feature to a value boundary
      if (threadIdx.x > 0) {        
        lastival = tmp;
        lastimpty = tmpx;
      }

      if (ival == lastival) lastimpty = 1e7f;               // Eliminate values which are not at value boundaries
      if (lastimpty < minimpty) {
        minimpty = lastimpty;
        bestival = ival;
      }
      minup2(minimpty,bestival);
      minimpty = __shfl(minimpty, jlast);                   // Carefully copy the last active thread to all threads, needed outside this loop
      bestival = __shfl(bestival, jlast);
      ctot = __shfl(ctot, jlast);
      cacc = __shfl(cacc, jlast);
      cact = __shfl(cact, jlast);
      lastival = __shfl(ival, jlast);
      lastimpty = __shfl(impty, jlast);
    }
    if (threadIdx.x == 0) {
      outv[i] = bestival;                                   // Output the best split feature value
      outf[i] = ((int)(key >> ishift)) & imask;             // Save the feature index
      outg[i] = T::fresult(caccall, ctotall) - minimpty;    // And the impurity gain
      outc[i] = imaxcnt;
    }
  }
}

template<typename T>
__global__ void __minImpurityb(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens, 
                              int nnodes, int ncats, int nsamps) {
  __shared__ int catcnt[DBSIZE];
  __shared__ int cattot[DBSIZE/4];
  __shared__ int stott[32];
  __shared__ float sacct[32];
  __shared__ int slastival[64];
  __shared__ int sbestival[32];
  __shared__ float sminimpty[32];
  int tid = threadIdx.x + blockDim.x * threadIdx.y;

  if (tid < 6) {
    catcnt[tid] = fieldlens[tid];
  }
  __syncthreads();
  int vshift = catcnt[5];
  int ishift = catcnt[4] + vshift;

  int cmask = (1 << catcnt[5]) - 1;
  int vmask = (1 << catcnt[4]) - 1;
  int imask = (1 << catcnt[3]) - 1;

  __syncthreads();

  int i, j, k, h, jc0, jc1, ilast, jlast;
  long long key;
  int cold, tot, ctt, tott, cnew, cnt, ncnt, tcnt, ival, icat, lastival, bestival, tmp;
  float update, updatet, acc, acct, impty, minimpty;

  for (i = blockIdx.x; i < nnodes*nsamps; i += gridDim.x) {
    // Process a group with fixed itree, inode, and ifeat

    jc0 = jc[i];                                              // The range of indices for this group
    jc1 = jc[i+1];
    __syncthreads();

    // Clear the cat counts and totals
    for (j = threadIdx.x; j < ncats; j += blockDim.x) {
      catcnt[j + threadIdx.y * blockDim.x] = 0;
      if (threadIdx.y == 0) cattot[j] = 0;
    }
    if (threadIdx.y == 0) {
      sminimpty[threadIdx.x] = 1e7f;
      sbestival[threadIdx.x] = -1;
    }
    __syncthreads();
    // First pass gets counts for each category and the (ci)log(ci) sum for this entire ifeat group

    for (j = jc0; j < jc1; j += blockDim.x * blockDim.x) {
      if (j + tid < jc1) {                                    // Read a block of keys and counts
        key = keys[j + tid]; 
        cnt = counts[j + tid];
        icat = ((int)key) & cmask;                            // Extract the cat id 
        atomicAdd(&cattot[icat + threadIdx.y * ncats], cnt);                        // Update count totals 
      }
    }

    __syncthreads();
    
    tott = 0;                                                 // Compute total count and (c)log(c) for the entire ifeat group
    acct = 0;
    if (threadIdx.y == 0) {
      for (k = 0; k < ncats; k += blockDim.x) {
        if (k + threadIdx.x < ncats) {
          tcnt = cattot[k + threadIdx.x];
          update = T::fupdate(tcnt);
        } else {
          tcnt = 0;
          update = 0;
        }
        accumup2(tcnt,update);
        ilast = min(31, ncats - k - 1);
        tcnt = __shfl(tcnt, ilast);
        update = __shfl(update, ilast);
        tott += tcnt;
        acct += update;
      }
      stott[threadIdx.x] = tott;
      sacct[threadIdx.x] = acct;
    }
    tott = stott[threadIdx.x];
    //    if (tid == 0 && i < 32) printf("cuda %d %d %f\n", i, tott, acct);

    // Main loop, work on blocks of 1024 (ideally)
    

    for (j = jc0; j < jc1; j += blockDim.x * blockDim.x) {

      for (k = 0; k < ncats; k += blockDim.x) {               // copy cumcounts from last row of last iteration to the first row
        tmp = catcnt[k + threadIdx.x + (blockDim.y -1) * ncats];
        __syncthreads();
        if (threadIdx.y == 0) {
          catcnt[k + threadIdx.x] = tmp;
        } else {
          catcnt[k + threadIdx.x + threadIdx.y * ncats] = 0;
        }
        __syncthreads();
      }

      if (j + tid < jc1) {                                    // Read a block of keys and counts
        key = keys[j + tid]; 
        cnt = counts[j + tid];
        icat = ((int)key) & cmask;                            // Extract the cat id and integer value;
        ival = ((int)(key >> vshift)) & vmask;                
        atomicAdd(&catcnt[icat + threadIdx.y * ncats], cnt);  // Update count totals 
      }
      jlast = min(31, jc1 - j - threadIdx.y * 32 - 1);        // Save the last value in this group
      if (threadIdx.x == jlast) {
        slastival[threadIdx.y + 1] = ival;
      }

      __syncthreads();

      for (k = 0; k < ncats; k += blockDim.x) {               // Form the cumsum along columns of catcnts
        for (h = 1; h < blockDim.y; h = h + h) {
          if (k + threadIdx.x < ncats && blockIdx.y + h < blockDim.y) {
            tmp = catcnt[k + threadIdx.x + ncats * threadIdx.y];
          }
          __syncthreads();
          if (k + threadIdx.x < ncats && blockIdx.y + h < blockDim.y) {
            catcnt[k + threadIdx.x + ncats * (threadIdx.y + h)] += tmp;
          }
          __syncthreads();
        }        
      }  

      tot = 0;                                                 // Local to a yblock (row) of catcnts                                                
      acc = 0.0f; 
      acct = 0.0f;
      for (k = 0; k < ncats; k += blockDim.x) {                // Now sum within a row (yblock)
        if (k + threadIdx.x < ncats) {
          cnt = catcnt[k + threadIdx.x + threadIdx.y * ncats];
          update = T::fupdate(cnt);
          updatet = T::fupdate(cattot[k + threadIdx.x] - cnt);
        } else {
          cnt = 0;
          update = 0;
          updatet = 0;
        }
        accumup3(cnt,update,updatet);
        ilast = min(31, ncats - k - 1);
        update = __shfl(update, ilast);
        updatet = __shfl(updatet, ilast);
        cnt = __shfl(cnt, ilast);
        tot += cnt;
        acc += update;
        acct += updatet;
      }

      __syncthreads();
    
      // OK, we have everything needed now to compute impurity for the rows in this yblock: 
      // tot, acc, acct at the end of the block

      lastival = -1;
      minimpty = 1e7f;

      ncnt = -cnt;
      for (k = jlast; k >= 0; k--) {                           // Sequentially update counts so that each thread
        if (threadIdx.x == k) {                                // in this warp gets the old and new counts
          cold = catcnt[icat + ncats * threadIdx.y];           // i.e. data for item k is in thread k
          ctt = cattot[icat + ncats * threadIdx.y];  
          cnew = cold + ncnt;
          catcnt[icat + ncats * threadIdx.y] = cnew;
        }
      }
      update = T::fupdate(cnew) - T::fupdate(cold);
      updatet = T::fupdate(ctt - cnew) - T::fupdate(ctt - cold);

      accumdown3(ncnt,update,updatet,jlast);
      tot += cnt;                                              // Now update the total c and total ci log ci sums
      acc += update;
      acct += updatet;
    
      impty = T::fresult(acc, tot) + T::fresult(acct, tott - tot); // And the impurity for this input

      tmp = __shfl_up(ival, 1);
      if (threadIdx.x > 0) {                                  // Get the last ival to check for a boundary
        lastival = tmp;
      } else {
        lastival = slastival[threadIdx.y];
      }
      __syncthreads();
      if (tid == 0) {
        tmp = slastival[33];
        slastival[0] = tmp;
      }
      __syncthreads();

      if (ival == lastival) impty = 1e7f;                    // Eliminate values which are not at value boundaries
      if (impty < minimpty) {
        minimpty = impty;
        bestival = ival;
      }

      minup2(minimpty,bestival);

      minimpty = __shfl(minimpty, jlast);                
      bestival = __shfl(bestival, jlast);
      if (threadIdx.x == 0) {
        sminimpty[threadIdx.y] = minimpty;
        sbestival[threadIdx.y] = bestival;
      }
      __syncthreads();

      if (threadIdx.y == 0) {
        minimpty = sminimpty[threadIdx.x];
        bestival = sbestival[threadIdx.x];
        minup2(minimpty,bestival);
        minimpty = __shfl(minimpty, blockDim.y - 1);
        bestival = __shfl(bestival, blockDim.y - 1);
        sminimpty[threadIdx.x] = minimpty;
        sbestival[threadIdx.x] = bestival;
      }
      __syncthreads();
    }
    

    if (tid == 0) {
      outv[i] = bestival;                                    // Output the best split feature value
      outf[i] = (int)((key >> ishift) & imask);              // Save the feature index
      //      outg[i] = T::fresult(sacct[0], tott) - minimpty;   // And the impurity gain
      outg[i] = T::fresult(sacct[0], tott);   // And the impurity gain
    }
    __syncthreads();
  }
}
#else
template<class T>
__global__ void __minImpuritya(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens, 
                               int nnodes, int ncats, int nsamps) {}

template<class T>
__global__ void __minImpurityb(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens, 
                               int nnodes, int ncats, int nsamps) {}
#endif

int minImpurity(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens, 
                int nnodes, int ncats, int nsamps, int impType) {
  // Note: its safe to round ncats up to a multiple of 32, since its only used to split shmem
  int ny = min(32, DBSIZE/ncats/2);
  dim3 tdim(32, ny, 1);
  int ng = min(64, nnodes*nsamps);
  if ((impType & 2) == 0) {
    if ((impType & 1) == 0) {
      __minImpuritya<entImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, outc, jc, fieldlens, nnodes, ncats, nsamps);
    } else {
      __minImpuritya<giniImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, outc, jc, fieldlens, nnodes, ncats, nsamps);
    }
  } else {
    if ((impType & 1) == 0) {
      __minImpurityb<entImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, outc, jc, fieldlens, nnodes, ncats, nsamps);
    } else {
      __minImpurityb<giniImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, outc, jc, fieldlens, nnodes, ncats, nsamps);
    }
  }
  fflush(stdout);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void __findBoundaries(long long *keys, int *jc, int n, int njc, int shift) {
  __shared__ int dbuff[1024];
  int i, j, iv, lasti;

  int imin = ((int)(32 * ((((long long)n) * blockIdx.x) / (gridDim.x * 32))));
  int imax = min(n, ((int)(32 * ((((long long)n) * (blockIdx.x + 1)) / (gridDim.x * 32) + 1))));

  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  if (tid == 0 && blockIdx.x == 0) {
    jc[0] = 0;
  }
  __syncthreads();
  lasti = 0x7fffffff;
  for (i = imin; i <= imax; i += blockDim.x * blockDim.y) {
    iv = njc;
    if (i + tid < imax) {
      iv = (int)(keys[i + tid] >> shift);
      dbuff[tid] = iv;
    }
    __syncthreads();
    if (i + tid < imax || i + tid == n) {
      if (tid > 0) lasti = dbuff[tid - 1];
      if (iv > lasti) {
        for (j = lasti+1; j <= iv; j++) {
          jc[j] = i + tid;
        }
      }
      if (tid == 0) {
        lasti = dbuff[blockDim.x * blockDim.y - 1];
      }
    }
    __syncthreads();
    }
}


int findBoundaries(long long *keys, int *jc, int n, int njc, int shift) {
  int ny = min(32, 1 + (n-1)/32);
  dim3 tdim(32, ny, 1);
  int ng = min(64, 1+n/32/ny);
  __findBoundaries<<<ng,tdim>>>(keys, jc, n, njc, shift);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

template<typename T>
__global__ void __mergeIndsP1(T *keys, int *cspine, T *ispine,  T *vspine, int n) {
  __shared__ T dbuff[1024];
  int i, j, itodo, doit, total;
  T thisval, lastval, endval, tmp;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int imin = (int)(((long long)n) * blockIdx.x / gridDim.x);
  int imax = (int)(((long long)n) * (blockIdx.x + 1) / gridDim.x);
  
  total = 0;
  if (tid == 0) {
    lastval = keys[imin];
    ispine[blockIdx.x] = lastval;
  }
  for (i = imin; i < imax; i += blockDim.x * blockDim.y) {
    itodo = min(blockDim.x * blockDim.y, imax - i);
    __syncthreads();
    if (i + tid < imax) {
      thisval = keys[i + tid];
      dbuff[tid] = thisval;
    }
    __syncthreads();
    if (tid > 0 && i + tid < imax) lastval = dbuff[tid - 1];
    if (tid == 0) endval = dbuff[itodo-1];
    __syncthreads();
    if (i + tid < imax) {
      dbuff[tid] = (thisval == lastval) ? 0 : 1;
    }
    __syncthreads();
    for (j = 1; j < itodo; j = j + j) {
      doit = tid + j < itodo && (tid & ((j + j)-1)) == 0;
      if (doit) {
        tmp = dbuff[tid] + dbuff[tid + j];
      }
      __syncthreads();
      if (doit) {
        dbuff[tid] = tmp;
      }
      __syncthreads();
    }
    if (tid == 0) {
      total += dbuff[0];
      lastval = endval;
    }
    __syncthreads();
  }	   
  if (tid == 0) {
    cspine[blockIdx.x] = total;
    vspine[blockIdx.x] = endval;
  } 
}

template<typename T>
__global__ void __fixSpine(int *cspine, T *ispine, T *vspine, int n) {
  __shared__ int counts[1024];
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int i, tmp;

  if (tid < n) {
    counts[tid] = cspine[tid];
  }
  __syncthreads();
  if (tid < n - 1) {
    if (ispine[tid + 1] != vspine[tid]) {
      counts[tid] += 1;
    }
  }
  __syncthreads();
  for (i = 1; i < n; i = i << 1) {
    if (tid >= i) {
      tmp = counts[tid - i];
    }
    __syncthreads();
    if (tid >= i) {
      counts[tid] += tmp;
    }
    __syncthreads();
  }
  if (tid == 0) {
    counts[n-1] += 1;
  }
  __syncthreads();
  if (tid < n) {
    cspine[tid] = counts[tid];
  }
}
    
template<typename T>
__global__ void __mergeIndsP2(T *keys, T *okeys, int *counts, int *cspine, int n) {
  __shared__ T dbuff[1024];
  __shared__ T obuff[2048];
  __shared__ int ocnts[2048];
  __shared__ int icnts[1024];
  int i, j, itodo, doit, lastcnt, lastocnt, obase, odone, total, coff;
  T thisval, lastval, tmp;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int imin = (int)(((long long)n) * blockIdx.x / gridDim.x);
  int imax = (int)(((long long)n) * (blockIdx.x + 1) / gridDim.x);
  int nbthreads = blockDim.x * blockDim.y;
  
  if (blockIdx.x == 0) {
    odone = 0;
  } else {
    odone = cspine[blockIdx.x - 1];
  }
  obase = 0;
  lastocnt = imin;
  if (tid == 0) {
    lastval = keys[imin];
  }
  for (i = imin; i < imax; i += nbthreads) {
    itodo = min(nbthreads, imax - i);
    __syncthreads();
    if (i + tid < imax) {                                       // Copy a block of input data into dbuff
      thisval = keys[i + tid];
      dbuff[tid] = thisval;
    }
    __syncthreads();
    if (tid > 0 && i + tid < imax) lastval = dbuff[tid - 1];
    __syncthreads();
    if (i + tid < imax) {
      icnts[tid] = (thisval == lastval) ? 0 : 1;                // Bit that indicates a change of index
    }
    __syncthreads();
    for (j = 1; j < itodo; j = j << 1) {                        // Cumsum of these bits = where to put key
      doit = tid + j < itodo;
      if (doit) {
        tmp = icnts[tid] + icnts[tid + j];
      }
      __syncthreads();
      if (doit) {
        icnts[tid + j] = tmp;
      }
      __syncthreads();
    }
    total = icnts[itodo-1];
    __syncthreads();
    if (i + tid < imax && thisval != lastval) {                 // and save the key/counts there in buffer memory
      if (tid > 0) {
        lastcnt = icnts[tid-1];
      } else {
        lastcnt = 0;
      }
      obuff[obase + lastcnt] = lastval;
      ocnts[obase + lastcnt] = i + tid;
    }
    __syncthreads();
    obase += total;
    if (obase >= nbthreads) {                                   // Buffer full so flush it
      okeys[odone+tid] = obuff[tid];
      if (tid > 0) lastocnt = ocnts[tid-1];
      coff = ocnts[tid] - lastocnt;
      atomicAdd(&counts[odone+tid], coff); 
      lastocnt = ocnts[nbthreads-1];
      odone += nbthreads;
    }
    __syncthreads();
    if (obase >= nbthreads) {                                   // Copy top to bottom of buffer
      obuff[tid] = obuff[tid+nbthreads];
      ocnts[tid] = ocnts[tid+nbthreads];
      obase -= nbthreads;
    }
    __syncthreads();
  }	   
  if (tid == itodo-1) {
    obuff[obase] = thisval;
    ocnts[obase] = i - nbthreads + tid + 1;
  }
  __syncthreads();
  if (tid <= obase) {                                            // Flush out anything that's left
    okeys[odone+tid] = obuff[tid];
    if (tid > 0) lastocnt = ocnts[tid-1];
    coff = ocnts[tid] - lastocnt;
    atomicAdd(&counts[odone+tid], coff); 
  }
}

//
// Accepts an array of int64 keys which should be sorted. Outputs an array okeys with unique copies of each key,
// with corresponding counts in the *counts* array. cspine is a working storage array in GPUmem which should be
// passed in. The size of cspine should be at least nb32 * 32 bytes with nb32 as below (maximum 2048 bytes). 
// Returns the length of the output in cspine[0].
//
int mergeInds(long long *keys, long long *okeys, int *counts, int n, int *cspine) {
  cudaError_t err;
  int nthreads = min(n, 1024);
  int nt32 = 32*(1 + (nthreads-1)/32);
  int nblocks = min(1 + (n-1)/nthreads, 64);
  int nb32 = 32*(1+(nblocks - 1)/32);
  long long *ispine = (long long *)&cspine[2*nb32];
  long long *vspine = (long long *)&cspine[4*nb32];

  __mergeIndsP1<long long><<<nblocks,nt32>>>(keys, cspine, ispine, vspine, n);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err == 0) {
    __fixSpine<long long><<<1,nblocks>>>(cspine, ispine, vspine, nblocks);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
  }
  if (err == 0) {
    __mergeIndsP2<long long><<<nblocks,nt32>>>(keys, okeys, counts, cspine, n);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
  }
  if (err == 0) {
    cudaMemcpy(cspine, &cspine[nblocks-1], 4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
  }
  return err;
}
//
// Support function for mergeInds. Returns the length of the output arrays in cspine[0].
// cspine is a working storage array in GPUmem which should be passed in. 
// The size of cspine should be at least nb32 * 32 bytes with nb32 as below (maximum 2048 bytes). 
//

int getMergeIndsLen(long long *keys, int n, int *cspine) {
  cudaError_t err;
  int nthreads = min(n, 1024);
  int nt32 = 32*(1 + (nthreads-1)/32);
  int nblocks = min(1 + (n-1)/nthreads, 64);
  int nb32 = 32*(1+(nblocks - 1)/32);
  long long *ispine = (long long *)&cspine[2*nb32];
  long long *vspine = (long long *)&cspine[4*nb32];

  __mergeIndsP1<long long><<<nblocks,nt32>>>(keys, cspine, ispine, vspine, n);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err == 0) {
    __fixSpine<long long><<<1,nblocks>>>(cspine, ispine, vspine, nblocks);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
  }
  if (err == 0) {
    cudaMemcpy(cspine, &cspine[nblocks-1], 4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
  }
  return err;
}

__global__ void __floatToInt(int n, float *in, int *out, int nbits) {
  const int signbit = 0x80000000;
  const int mag =     0x7fffffff;
  int fshift = 32 - nbits;
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    float v = in[i];
    int vi = *((int *)&v);
    if (vi & signbit) {
      vi = -(vi & mag);
    }
    int ival = ((unsigned int)vi) >> fshift;
    out[i] = ival;
  }
}

int floatToInt(int n, float *in, int *out, int nbits) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __floatToInt<<<griddims,nthreads>>>(n, in, out, nbits);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void __jfeatsToIfeats(int itree, int *inodes, int *jfeats, int *ifeats, int n, int nfeats, int seed) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    int inode = inodes[i];
    int jfeat = jfeats[i];
    int ifeat = mmhash3(itree, inode, jfeat, nfeats, seed);
    ifeats[i] = ifeat;
  }
}

int jfeatsToIfeats(int itree, int *inodes, int *jfeats, int *ifeats, int n, int nfeats, int seed) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __jfeatsToIfeats<<<griddims,nthreads>>>(itree, inodes, jfeats, ifeats, n, nfeats, seed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}
