#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

#if __CUDA_ARCH__ >= 300

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


#define DBSIZE (5*1024)

__global__ void __multinomial2(int nrows, int ncols, float *A, int *B, curandState *rstates, int nvals) {
  __shared__ float vec[DBSIZE];
  __shared__ float tvec[DBSIZE];
  int i, j, jcol, jcolnr, jbase, shift, *ivec;
  int imin, imax, imid;
  int fitcols = DBSIZE/nrows;
  float vv, rval, *avec, *bvec, *cvec;
  float nrowsinv = 1.0f / nrows;
  float nvalsinv = 1.0f / nvals;
  curandState *prstate = &rstates[threadIdx.x];
  __syncthreads();
  for (i = fitcols * blockIdx.x; i < ncols; i += fitcols * gridDim.x) {
    avec = vec;
    bvec = tvec;
    __syncthreads();
    for (j = threadIdx.x; j < nrows * fitcols; j += blockDim.x) {
      vec[j] = A[j + i * nrows];
    }
    __syncthreads();
    for (shift = 1; shift < nrows; shift *= 2) {
      for (j = threadIdx.x; j < nrows * fitcols; j += blockDim.x) {
        vv = avec[j];
        jbase = ((int)floor((j + 0.5f)*nrowsinv))*nrows;
        //        jbase = (j / nrows) * nrows;
        if (j - shift >= jbase) {
          vv += avec[j-shift];
        }
        bvec[j] = vv;
      }
      __syncthreads();
      cvec = avec;
      avec = bvec;
      bvec = cvec;
    }
    ivec = (int *)bvec;
    for (j = threadIdx.x; j < nrows*fitcols; j += blockDim.x) {
      ivec[j] = 0;
    }
    __syncthreads();
    for (j = threadIdx.x; j < nvals*fitcols; j += blockDim.x) {
      jcol = (int)floor((j + 0.5f)*nvalsinv);
      jcolnr = jcol * nrows;
      rval = avec[jcolnr+nrows-1]*curand_uniform(prstate);
      imin = 0;
      imax = nrows;
      while (imax - imin > 1) {
        imid = (imin + imax) >> 1;
        if (rval >= avec[imid + jcolnr]) {
          imin = imid;
        } else {
          imax = imid;
        }
      }
      atomicAdd(&ivec[imin + jcolnr], 1);
    }
    __syncthreads();
    for (j = threadIdx.x; j < nrows*fitcols; j += blockDim.x) {
      B[j + i * nrows] = ivec[j];
    }
    __syncthreads();
  } 
}       

//
// 

__forceinline__ __device__ int __waittimeval(curandState *prstate, float p, int n) {
  float q = - log(1-p);
  float X = 0;
  float sum = 0;
  int i = 0;
  while (i < 100 && sum <= q) {
    float E = - log(curand_uniform(prstate));  // safe since curand_uniform wont return 0
    sum += E / (n - X);
    X += 1;
    i += 1;
  }
  return X - 1;  
}

__forceinline__ __device__ int binorndval(float p, int n, curandState *prstate) {
  bool pflipped;
  float X, Y, V;
  const float pi = 3.1415926f;

  if (p > 0.5f) {                            // flip p so that its less than 1/2.
    pflipped = true;
    p = 1.0f - p;
  } else {
    pflipped = false;
  }
  float np = n * p;
  if (np < 21) {
    X = __waittimeval(prstate, p, n);           // Use a wait time method if small expected output
  } else {
    float oldp = p;                   
    p = floor(np) / n;                       // round np to an integral value for the rejection stage
    p = max(1e-7f, min(1 - 1e-7f, p));       // prevent divide-by-zeros
    np = n * p;
    float n1mp = n * (1-p);
    float pvar = np * (1-p);
    float delta1 = max(1.0f, floor(sqrt(pvar * log(128 * np / (81 * pi * (1-p))))));
    float delta2 = max(1.0f, floor(sqrt(pvar * log(128 * n1mp / (pi * p)))));
    float sigma1 = sqrt(pvar)*(1+delta1/(4*np));
    float sigma2 = sqrt(pvar)*(1+delta2/(4*n1mp));
    float sigma1sq = sigma1 * sigma1;
    float sigma2sq = sigma2 * sigma2;
    float c = 2 * delta1 / np;
    float a1 = 0.5f * exp(c) * sigma1 * sqrt(2*pi);
    float a2 = 0.5f * sigma2 * sqrt(2*pi);
    float a3 = exp(delta1/n1mp - delta1*delta1/(2*sigma1sq))*2*sigma1sq/delta1;
    float a4 = exp(-delta2*delta2/(2*sigma2sq))*2*sigma2sq/delta2;
    float s = a1 + a2 + a3 + a4;
    int i = 0;
    while (i < 100) {                            // Give up eventually
      i += 1;
      float U = s * curand_uniform(prstate);
      float E1 = - log(curand_uniform(prstate)); // safe since curand_uniform wont return 0
      if (U <= a1 + a2) {
        float N = curand_normal(prstate);
        if (U <= a1) {
          Y = sigma1 * abs(N);
          if (Y >= delta1) continue;
          X = floor(Y);
          V = - E1 - N * N/2 + c;
        } else {
          Y = sigma2 * abs(N);
          if (Y >= delta2) continue;
          X = floor(-Y);
          V = - E1 - N * N/2;
        }
      } else {
        float E2 = - log(curand_uniform(prstate));
        if (U <= a1 + a2 + a3) {
          Y = delta1 + 2*sigma1sq*E1/delta1;
          X = floor(Y);
          V = - E2 - delta1*Y/(2*sigma1sq) + delta1/n1mp;
        } else {
          Y = delta2 + 2*sigma2sq*E1/delta2;
          X = floor(-Y);
          V = - E2 - delta2*Y/(2*sigma2sq);
        }
      }
      if (X < - np || X > n1mp) continue;
      if (V > lgamma(np+1) + lgamma(n1mp+1) - lgamma(np+X+1) - lgamma(n1mp-X+1) + X*log(p/(1-p))) continue;
      break;
    }
    X += np;
    X += __waittimeval(prstate, (oldp-p)/(1-p), n-X); // Now correct for rounding np to integer
  }
  if (pflipped) {                                  // correct for flipped p. 
    X = n - X;
  }
  return (int)X;
}

//
// i steps over blocks of 256 columns
//   j steps down blocks of 32 rows
//     Load a block of 32 x 256 words
//     k steps down the rows of this block
//       compute local p get bino sample "count" from remaining n
//       decrement remaining n. 
//

__global__ void __multinomial(int nrows, int ncols, float *A, int *B, float *Norm, curandState *rstates, int nvals) {
  __shared__ float mat[256][33];

  int (*imat)[33] = (int (*)[33])mat;
  int i, j, k, valsleft, count, iv;
  float vnorm, vv;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  curandState *prstate = &rstates[tid + blockIdx.x*blockDim.x*blockDim.y];
  __syncthreads();
  for (i = 256*blockIdx.x; i < ncols; i += 256*gridDim.x) {   // Loop across blocks of 256 columns
    vnorm = 1.0f;
    if (tid + i < ncols) {                                    // Load the norms for these 256 cols
      vnorm = Norm[tid+i];
    }
    valsleft = nvals;                                         // Initialize the count of samples for this column
    for (j = 0; j < nrows; j += blockDim.x) {                 // Loop over blocks of 32 rows
      __syncthreads();
      for (k = 0; k < min(256, ncols-i); k += 8) {            // Copy this 32x256 word block into SHMEM
        vv = 0;
        if (j+threadIdx.x < nrows) {
          vv = A[j+threadIdx.x + (i+k+threadIdx.y)*nrows];
        }
        mat[k+threadIdx.y][threadIdx.x] = vv;
      }
      __syncthreads();
      for (k = 0; k < min(32, nrows-j); k += 1) {             // Now walk down the columns with 256 threads
        vv = min(mat[tid][k], vnorm);
        count = binorndval(vv/vnorm, valsleft, prstate);      // get a binomial random count
        count = min(count, valsleft);
        valsleft -= count;                                    // subtract it from remaining count
        vnorm -= vv;                                          // adjust remaining probability
        imat[tid][k] = count;                                 // store count in the aliased SHMEM matrix
      }
      __syncthreads();
      for (k = 0; k < min(256, ncols-i); k += 8) {            // Save this 32x256 block back into main memory
        iv = imat[k+threadIdx.y][threadIdx.x];
        if (j+threadIdx.x < nrows) {
          B[j+threadIdx.x + (i+k+threadIdx.y)*nrows] = iv;
        }
      }
      __syncthreads();
    }
  } 
}       

__global__ void __prandinit(curandState *rstates) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(1234, id, 0, &rstates[id]);
}

int multinomial2(int nrows, int ncols, float *A, int *B, int nvals) {
  int fitcols = DBSIZE/nrows;
  int nthreads = 1024;
  int nb =  1 + (ncols-1)/fitcols;
  int nblocks = min(128, nb);
  curandState *rstates;
  cudaError_t err = cudaMalloc(( void **)& rstates , nblocks * nthreads * sizeof(curandState));
  if (err > 0) {
    fprintf(stderr, "Error in cudaMalloc %d", err);
    return err;
  }
  cudaDeviceSynchronize();
  __prandinit<<<nblocks,nthreads>>>(rstates); 
  cudaDeviceSynchronize();
  __multinomial2<<<nblocks,nthreads>>>(nrows, ncols, A, B, rstates, nvals);
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}

int multinomial(int nrows, int ncols, float *A, int *B, float *Norm, int nvals) {
  dim3 threads(32, 8, 1);
  int nthreads = 256;
  int nblocks = min(128, 1+ (ncols-1)/256);
  curandState *rstates;
  cudaError_t err = cudaMalloc(( void **)& rstates , nblocks * nthreads * sizeof(curandState));
  if (err > 0) {
    fprintf(stderr, "Error in cudaMalloc %d", err);
    return err;
  }
  cudaDeviceSynchronize();
  __prandinit<<<nblocks,nthreads>>>(rstates); 
  cudaDeviceSynchronize();
  __multinomial<<<nblocks,threads>>>(nrows, ncols, A, B, Norm, rstates, nvals);
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}

template<typename KEY, typename V1, typename V2, typename RET, class C>
  __global__ void prodSelect(int n, int *groups, KEY *keys1, KEY *keys2, V1 *vals1, V2 *vals2, KEY *kout, RET *ret) {
}
