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
        long long hdr = (((long long)(tmask & itree)) << tshift) | (((long long)(nmask & inode)) << nshift) | 
          (((long long)(jmask & jfeat)) << jshift) | (((long long)(imask & ifeat)) << ishift) ;
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
  static __device__ inline float fupdate(int v) { return (float)v * log((float)max(1, v)); }
  static __device__ inline float fresult(float vacc, int vsum) { float vs = (float)max(1, vsum); return log(vs) - vacc / vs; }
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

__device__ inline void accumup3(int &cnt, float update, float &updatet) {
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

template<typename T>
__global__ void __minImpuritya(long long *keys, int *counts, int *outv, int *outf, float *outg, int *jc, int *fieldlens,
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

  int i, j, k, h, jc0, jc1, jlast;
  long long key;
  int cold, ctot, ctot2, ctt, ctotall, cnew, cnt, ival, icat, lastival, bestival, tmp;
  float update, updatet, cacc, cact, impty, minimpty, lastimpty, tmpx, tmpy;

  printf("cuda %d\n", tid);

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
      cnt = __shfl(cnt, jlast);                             // Now update the total c and total ci log ci sums
      update = __shfl(update, jlast);
      ctot += cnt;
      cacc += update;
    }
    __syncthreads();
    if (threadIdx.x == 0 && i < 32) printf("cuda %d %d %f\n", i, ctot, cacc);

    // Second pass to compute impurity at every input point
    cact = cacc;                                            // Save the total count and (ci)log(ci) sum
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
        icat = (int)(key & cmask);                          // Extract the cat id and integer value
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
      impty = T::fresult(cacc, ctot) + T::fresult(cact, ctotall - ctot); // And the impurity for this input

      tmp = __shfl_up(ival, 1);                             // Need the last impurity and ival in order
      tmpx = __shfl_up(impty, 1);                           // to restrict the partition feature to a value boundary
      if (threadIdx.x > 0) {        
        lastival = tmp;
        lastimpty = tmpx;
      }

      if (ival == lastival) lastimpty = 1e7f;               // Eliminate values which are not at value boundaries
      if (lastimpty < minimpty) {
        minimpty = lastimpty;
        bestival = lastival;
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
      outf[i] = (int)((key >> ishift) & imask);             // Save the feature index
      outg[i] = T::fresult(cacc, ctot) - minimpty;          // And the impurity gain
    }
  }
}

template<typename T>
__global__ void __minImpurityb(long long *keys, int *counts, int *outv, int *outf, float *outg, int *jc, int *fieldlens, 
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
        atomicAdd(&cattot[icat], cnt);                        // Update count totals 
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

    // Main loop, work on blocks of 1024 (ideally)
    /*

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
    */

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
__global__ void __minImpuritya(long long *keys, int *counts, int *outv, int *outf, float *outg, int *jc, int *fieldlens, 
                              int nnodes, int ncats, int nsamps) {}

template<class T>
__global__ void __minImpurityb(long long *keys, int *counts, int *outv, int *outf, float *outg, int *jc, int *fieldlens, 
                              int nnodes, int ncats, int nsamps) {}
#endif

int minImpurity(long long *keys, int *counts, int *outv, int *outf, float *outg, int *jc, int *fieldlens, 
                int nnodes, int ncats, int nsamps, int impType) {
  // Note: its safe to round ncats up to a multiple of 32, since its only used to split shmem
  int ny = min(32, DBSIZE/ncats/2);
  dim3 tdim(32, ny, 1);
  int ng = min(64, nnodes*nsamps);
  printf("CUDA %d\n", impType);
  if (impType & 2 == 0) {
    if (impType & 1 == 0) {
      __minImpuritya<entImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, jc, fieldlens, nnodes, ncats, nsamps);
    } else {
      __minImpuritya<giniImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, jc, fieldlens, nnodes, ncats, nsamps);
    }
  } else {
    if (impType & 1 == 0) {
      __minImpurityb<entImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, jc, fieldlens, nnodes, ncats, nsamps);
    } else {
      __minImpurityb<giniImpty><<<ng,tdim>>>(keys, counts, outv, outf, outg, jc, fieldlens, nnodes, ncats, nsamps);
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
__global__ void mergeIndsP1(T *keys, int *cspine, T *ispine, T *vspine, int n) {
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
    itodo = min(1024, imax - i);
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
    for (j = 1; j < itodo; j = j << 1) {
      doit = tid + j < itodo && (tid & ((j << 1)-1)) == 0;
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
  }	   
  if (tid == 0) {
    cspine[blockIdx.x] = total;
    vspine[blockIdx.x] = endval;
  }
}

template<typename T>
__global__ void fixSpine(int *cspine, T *ispine, T *vspine, int n) {
  __shared__ int counts[1024];
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int i, tmp;

  if (tid < n) {
    counts[tid] = cspine[tid];
  }
  __syncthreads();
  if (tid < n - 1) {
    if (ispine[tid + 1] != vspine[tid]) {
      counts[tid + 1] += 1;
    }
  }
  if (tid == 0) {
    counts[0] += 1;
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
  if (tid < n) {
    cspine[tid] = counts[tid];
  }
}
    
template<typename T>
__global__ void mergeIndsP2(T *keys, T *okeys, int *counts, int *cspine, int n) {
  __shared__ T dbuff[1024];
  __shared__ T obuff[2048];
  __shared__ int ocnts[2048];
  int i, j, itodo, doit, thiscnt, lastcnt, obase, odone, total;
  T thisval, lastval, endval, tmp;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int imin = (int)(((long long)n) * blockIdx.x / gridDim.x);
  int imax = (int)(((long long)n) * (blockIdx.x + 1) / gridDim.x);
  
  odone = cspine[blockIdx.x];
  obase = 0;
  if (tid == 0) {
    lastval = keys[imin];
  }
  for (i = imin; i < imax; i += blockDim.x * blockDim.y) {
    itodo = min(1024, imax - i);
    __syncthreads();
    if (i + tid < imax) {                                // Copy a block of input data into dbuff
      thisval = keys[i + tid];
      dbuff[tid] = thisval;
    }
    __syncthreads();
    if (tid > 0 && i + tid < imax) lastval = dbuff[tid - 1];
    if (tid == 0) endval = dbuff[itodo-1];       
    __syncthreads();
    if (i + tid < imax) {
      ocnts[tid] = (thisval == lastval) ? 0 : 1;        // Bit that indicates a change of index
    }
    __syncthreads();
    for (j = 1; j < itodo; j = j << 1) {                // Cumsum of these bits = where to put key
      doit = tid + j < itodo && (tid & ((j << 1)-1)) == 0;
      if (doit) {
        tmp = ocnts[tid] + ocnts[tid + j];
      }
      __syncthreads();
      if (doit) {
        ocnts[tid] = tmp;
      }
      __syncthreads();
    }
    total = ocnts[itodo-1];
    if (tid > 0 && i + tid < imax) {                    // Find where the index changes
      thiscnt = ocnts[tid];
      lastcnt = ocnts[tid-1];
    }
    __syncthreads();
    if (tid > 0 && i + tid < imax) {                    // and save the key/counts there in buffer memory
      if (thiscnt > lastcnt) {
        obuff[obase + thiscnt] = thisval;
        ocnts[obase + thiscnt] = i + tid;
      }
    }
    __syncthreads();
    obase += total;
    if (obase > 1024) {                                 // Buffer full so flush it
      okeys[odone+tid] = obuff[tid];
      counts[odone+tid] = ocnts[tid] - ocnts[tid-1];    // Need to fix wraparound
      odone += 1024;
    }
    __syncthreads();
    if (obase > 1024) {                                 // Copy top to bottom of buffer
      obuff[tid] = obuff[tid+1024];
      ocnts[tid] = ocnts[tid+1024];
    }
    obase -= 1024;
  }	   
  if (tid < obase) {                                    // Flush out anything that's left
    okeys[odone+tid] = obuff[tid];
    counts[odone+tid] = ocnts[tid] - ocnts[tid-1];      // Need to fix wraparound
  }
}

