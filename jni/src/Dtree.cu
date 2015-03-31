#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>
#include <MurmurHash.hpp>

static const int dtsignbit = 0x80000000;
static const int dtmag =     0x7fffffff;

__forceinline__ __device__ int getFloatBits(float val, int fshift) {
  int ival = *((int *)&val);
  if (ival & dtsignbit) {
    ival = -(ival & dtmag);
  }
  ival += dtsignbit;
  ival = ((unsigned int)ival) >> fshift;
  return ival;
}

__forceinline__ __device__ int getFloatBits(int ival, int fshift) {
  return ival;
}

#define DBSIZE (8*1024)

// threadIdx.x is the feature index
// threadIdx.y is the tree index
// blockIdx.x and blockIdx.y index blocks of columns
template <typename S, typename T>
__global__ void __treePack(S *fdata, int *treenodes, T *icats, long long *out, int *fieldlens, 
                           int nrows, int ncols, int ntrees, int nsamps, int seed) {
  __shared__ S fbuff[DBSIZE];
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

  int fshift = 32 - fl[4];

  for (i = nc * blockIdx.x; i < ncols; i += nc * gridDim.x) {
    int ctodo = min(nc, ncols - i);
    for (j = tid; j < nrows * ctodo; j += blockDim.x*blockDim.y) {
      fbuff[j] = fdata[j + i * nrows];
    }
    __syncthreads();
    
    for (j = i; j < i + ctodo; j++) {                               // j is the column index
      ic = (int)icats[j];
      for (itree = threadIdx.y; itree < ntrees; itree += blockDim.y) {
        if (jfeat < nsamps) {
          int inode0 = treenodes[itree + j * ntrees];
          int inode = inode0 & 0x7fffffff;
          long long isign = ((long long)((inode0 & dtsignbit) ^ dtsignbit)) << 32;
          int ifeat = mmhash3(itree, inode, jfeat, nrows, seed);
          S v = fbuff[ifeat + (j - i) * nrows];
          int ival = getFloatBits(v, fshift);
          long long hdr = 
            (((long long)(tmask & itree)) << tshift) | (((long long)(nmask & inode)) << nshift) | 
            (((long long)(jmask & jfeat)) << jshift) | (((long long)(imask & ifeat)) << ishift) |
            (((long long)(vmask & ival)) << vshift) | ((long long)(ic & cmask)) | isign;
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
  __treePack<float,int><<<nb,bdim>>>(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

int treePackfc(float *fdata, int *treenodes, float *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed) {
  int ntx = 32 * (1 + (nsamps - 1)/32);
  int nty = min(1024 / ntx, ntrees);
  dim3 bdim(ntx, nty, 1);
  int nb = min(32, 1 + (ncols-1)/32);
  __treePack<float,float><<<nb,bdim>>>(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


int treePackInt(int *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed) {
  int ntx = 32 * (1 + (nsamps - 1)/32);
  int nty = min(1024 / ntx, ntrees);
  dim3 bdim(ntx, nty, 1);
  int nb = min(32, 1 + (ncols-1)/32);
  __treePack<int,int><<<nb,bdim>>>(fdata, treenodes, icats, out, fieldlens, nrows, ncols, ntrees, nsamps, seed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

// threadIdx.x is the tree index
// threadIdx.y is a column index
// blockIdx.x and blockIdx.y index blocks of columns

__global__ void __treeWalk(float *fdata, int *inodes, float *fnodes, int *itrees, int *ftrees, int *vtrees, float *ctrees,
                           int nrows, int ncols, int ntrees, int nnodes, int getcat, int nbits, int nlevels) {
  __shared__ float fbuff[DBSIZE];
  int i, j, k, itree, inode, ipos, ftree, vtree, ifeat, ichild, big;
  float ctree, feat;

  int nc = (DBSIZE / nrows);
  int fshift = 32 - nbits;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int bid = blockIdx.x + gridDim.x * blockIdx.y;
  int nblocks = gridDim.x * gridDim.y;
  int nthreads = blockDim.x * blockDim.y;

  for (i = nc * bid; i < ncols; i += nc * nblocks) {             // i is a global block column index
    int ctodo = min(nc, ncols - i);
    // Fill up the SHMEM buffer with nc columns from fdata
    __syncthreads();
    for (j = tid; j < nrows * ctodo; j += nthreads) {
      fbuff[j] = fdata[j + i * nrows];
    }
    __syncthreads();
    
    for (j = threadIdx.y; j < ctodo; j += blockDim.y) {          // j is the (local SHMEM) column index
      for (itree = threadIdx.x; itree < ntrees; itree += blockDim.x) { // itree indexes the trees
        inode = 0;                                               // points to the current node
        ipos = itree * nnodes;                                   // address in the tree arrays of this node     
        for (k = 0; k < nlevels; k++) {
          ichild = itrees[ipos];                                 // left child index
          vtree = vtrees[ipos];                                  // and threshold 
          if (vtree == -2) {                                     // non-splittable node, so mark inode
            inode = inode | dtsignbit;
          }
          if (ichild == 0 || vtree == -2) break;                 // this is a leaf, so break
          ftree = ftrees[ipos];                                  // otherwise get split feature index
          feat = fbuff[ftree + j * nrows];                       // get the feature pointed to 
          ifeat = getFloatBits(feat, fshift);
          big = ifeat > vtree;                                   // compare with the threshold
          inode = ichild + big;                                  // address of left child in the block
          ipos = inode + itree * nnodes;                         // address in the tree arrays of this node
        }
        if (getcat) {                                            // save the leaf node index or the label
          ctree = ctrees[ipos];
          fnodes[itree + (i + j) * ntrees] = ctree;
        } else {
          inodes[itree + (i + j) * ntrees] = inode;
        }
      }
    }
    __syncthreads();
  }
}

int treeWalk(float *fdata, int *inodes, float *fnodes, int *itrees, int *ftrees, int *vtrees, float *ctrees,
             int nrows, int ncols, int ntrees, int nnodes, int getcat, int nbits, int nlevels) {
  int nc = DBSIZE / nrows;
  int xthreads = min(ntrees,1024);
  int ythreads = min(nc,1024/xthreads);
  dim3 threaddims(xthreads, ythreads, 1);
  int nblocks = 1 + (ncols-1) / 8 / nc;
  int yblocks =  1 + (nblocks-1)/65536;
  int xblocks = 1 + (nblocks-1)/yblocks;
  dim3 blockdims(xblocks, yblocks, 1);
  //  printf("nrows %d, ncols %d, ntrees %d, nnodes %d, getcat %d, nbits %d, nlevels %d, xthreads %d, ythreads %d, xblocks %d, yblocks %d\n",
  //  nrows, ncols, ntrees, nnodes, getcat, nbits, nlevels, xthreads, ythreads, xblocks, yblocks);
  __treeWalk<<<blockdims,threaddims>>>(fdata, inodes, fnodes, itrees, ftrees, vtrees, ctrees, nrows, ncols, ntrees, nnodes, getcat, nbits, nlevels);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

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

__global__ void __floatToInt(int n, float *in, int *out, int nbits) {
  int fshift = 32 - nbits;
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    float v = in[i];
    int ival = getFloatBits(v, fshift);
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
