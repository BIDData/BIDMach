#include <cuda_runtime.h>
#include <stdio.h>
#include "MatKernel.hpp"

#if __CUDA_ARCH__ > 200

#define edcellupdate(RR,RP1,RP2,RPP,WUN,TMP)                                                               \
  asm("vmin4.s32.s32.s32.add" "%0, %1.b3210, %2.b4321, %3;": "=r" (RR) : "r" (RP1), "r" (RP2), "r" (WUN)); \
  asm("vadd4.s32.s32.s32" "%0, %1, %2, %3;": "=r" (TMP) : "r" (MM), "r" (RZ), "r" (RR));                   \
  asm("vmin4.s32.s32.s32" "%0, %1, %2, %3;": "=r" (RR) : "r" (TMP), "r" (RR), "r" (RR));       


__device__ void hammingcell(int &a0, int a1, int b0, int w0, int &c, int tmp, int zero) {
  asm("and.b32" "%0, %1, %2;": "=r" (tmp) : "r" (a0), "r" (b0));
  asm("vset4.s32.s32.eq" "%0, %1, %2, %3;": "=r" (tmp) : "r" (tmp), "r" (zero), "r" (zero));
  asm("vsub4.s32.s32.s32" "%0, %1, %2, %3;": "=r" (tmp) : "r" (zero), "r" (tmp), "r" (zero));
  asm("vmin4.u32.u32.u32.add" "%0, %1, %2, %3;": "=r" (c) : "r" (w0), "r" (tmp), "r" (c));
  asm("vmax4.u32.u32.u32" "%0, %1.b4321, %2.b4321, %3;": "=r" (a0) : "r" (a0), "r" (a1), "r" (a0));  
}

__device__ void rotate1(int &a0) {
  asm("shr.b32" "%0, %1, 8;": "=r" (a0) : "r" (a0)); 
}

__device__ void editcell(unsigned int a0, unsigned int a1, unsigned int m, unsigned int &b0, unsigned int &b1) {
  unsigned int a, am, c, nd, ne, f0, f1;
  a = a0 & ~ a1;              // a = 1
  am = a & m;
  c = (a + am) ^ a ^ am;      // carry bit
  nd = m | c | (a0 & a1);     // complement of diagonal bit d
  ne = nd >> 1;               // shifted diagonal bit

  f0 = nd ^ ne;               // f = bits of e - d
  f1 = ne & ~ nd;
  b0 = a0 ^ f0;
  b1 = (a1 & ~ f0) | (f1 & ~ a0) ; 
}

__device__ void shlc(unsigned int &a0, unsigned int &a1) {
  asm("add.cc.u32" "%0, %1, %2;": "=r" (a0) : "r" (a0), "r" (a0));
  asm("addc.cc.u32" "%0, %1, %2;": "=r" (a1) : "r" (a1), "r" (a1));
}

template<int VECLEN, int NVEC, int TLEN>
  __global__ void __hammingdists(int *a, int *b, int *w, int *op, int *ow, int n) {   
  __shared__ int sa[TLEN];
  __shared__ int sb[32][VECLEN*NVEC+1];
  __shared__ int sw[32][VECLEN*NVEC+1];
  __shared__ int sop[32];
  __shared__ int sow[32];
  register int aa[VECLEN+1];           
  register int bb[VECLEN];
  register int ww[VECLEN];
  int i, ioff, ioffmv, ip, tmp, tmp1, j, k, c, cmin, imin;
  int zero = 0;
  int sid = threadIdx.x + blockDim.x * threadIdx.y;

  if (threadIdx.y + blockDim.y * blockIdx.x < n) {

    // Load data into shared memory
    for (i = 0; i < TLEN/1024; i++) {
      sa[sid + i*1024] = a[sid + i*1024 + TLEN*blockIdx.x];
    }
    for (i = 0; i < VECLEN*NVEC/32; i++) {
      sb[threadIdx.y][threadIdx.x + i*blockDim.x] = b[sid + i*1024 + VECLEN*NVEC*blockIdx.x];
      sw[threadIdx.y][threadIdx.x + i*blockDim.x] = w[sid + i*1024 + VECLEN*NVEC*blockIdx.x];
    }
    __syncthreads();

    ip = threadIdx.x / NVEC;
    ioffmv = (threadIdx.x % NVEC) * VECLEN;
    ioff = ioffmv + ip * (TLEN*NVEC/32);
    cmin = 0x7fffffff;
    imin = -1;

    // Load data for this thread into registers
#pragma unroll
    for (j = 0; j < VECLEN; j++) {
      tmp = j + ioff;
      if (tmp < TLEN) {
        aa[j] = sa[tmp];
      }
      bb[j] = sb[threadIdx.y][j + ioffmv];
      ww[j] = sw[threadIdx.y][j + ioffmv];
    }
    // Step through offsets in A string
    for (j = 0; j < TLEN*NVEC/8; j++) {
      tmp = VECLEN + ioff + j / 4;
      if (tmp - ioffmv < TLEN - VECLEN * NVEC) {
        if (j % 4 == 0) {
          aa[VECLEN] = sa[tmp];
        }
        c = 0;
        // Inner loop over the length of the vector in registers
#pragma unroll
        for (k = 0; k < VECLEN; k++) {
          hammingcell(aa[k], aa[k+1], bb[k], ww[k], c, tmp, zero);
        }
        rotate1(aa[VECLEN]);
        // Need to sum over NVEC to get complete score for a string
#pragma unroll
        for (k = 1; k < NVEC; k *= 2) {    
          tmp = __shfl_down(c, k);  
          c = c + tmp;
        }
        // Now compare with the accumulated min
        if (c < cmin) {
          cmin = c;
          imin = 4 * ioff + j;
        }
      }
    }
    // Compute the min across groups of NVEC threads in this warp
    for (k = NVEC; k < 32; k *= 2) {    
      tmp = __shfl_down(cmin, k);
      tmp1 = __shfl_down(imin, k);
      if (tmp < cmin) {
        cmin = tmp;
        imin = tmp1;
      }
    }
    // Save to shared memory in prep for saving to main memory
    if (threadIdx.x == 0) {
      sop[threadIdx.y] = imin;
      sow[threadIdx.y] = cmin;
    }
    __syncthreads();
    // Save to main memory
    if (threadIdx.y == 0) {
      op[threadIdx.x + 32*blockIdx.x] = sop[threadIdx.x];
      ow[threadIdx.x + 32*blockIdx.x] = sow[threadIdx.x];
    }
  }
}

__global__ void __veccmp(int *a, int *b, int *d) {
  int xa = *a;
  int xb = *b;
  int xc = 0;
  int xd = 0;
  asm("vset4.s32.s32.ne" "%0, %1.b0000, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
  *d++ = xd;
  asm("vset4.s32.s32.ne" "%0, %1.b1111, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
  *d++ = xd;
  asm("vset4.s32.s32.ne" "%0, %1.b2222, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
  *d++ = xd;
  asm("vset4.s32.s32.ne" "%0, %1.b3333, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
  *d = xd;
}
#else
__global__ void __veccmp(int *a, int *b, int *d) {
  printf("__veccmp() not defined for CUDA Arch < 300\n");
}

template<int VECLEN, int NVEC, int TLEN>
__global__ void __hammingdists(int *a, int *b, int *w, int *op, int *ow, int n) {
  printf("__hammingdists() not defined for CUDA Arch < 300\n");
}
#endif

int veccmp(int *a, int *b, int *d) {
  __veccmp<<<1,1>>>(a, b, d);
  return 0;
}

int hammingdists(int *a, int *b, int *w, int *op, int *ow, int n) {    
  int nb = 1+((n-1)/32);
  dim3 blockdims(32,32,1);
  __hammingdists<16,2,1024><<<nb,blockdims>>>(a, b, w, op, ow, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}    
