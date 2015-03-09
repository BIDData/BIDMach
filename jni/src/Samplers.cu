#include <cuda_runtime.h>
#include <curand_kernel.h>
//#include <curand.h>
#include <stdio.h>
#include <MatKernel.hpp>

#define GBINODIMX 16
#define GBINOSIZE (8*1024/GBINODIMX)
#define GBINODIMY (GBINOSIZE/GBINODIMX)

// GBINOSIZE

#if __CUDA_ARCH__ >= 300

//
// This version creates k samples per input feature, per iteration (with a multinomial random generator).
// A and B are the factor matrices. Cir, Cic the row, column indices of the sparse matrix S, P its values, and nnz its size.
// S holds inner products A[:,i] with B[:,j]. Ms holds model samples, Us holds user samples. 
//
__global__ void __LDA_Gibbs1(int nrows, int nnz, float *A, float *B, int *Cir, int *Cic, float *P, int *Ms, int *Us, int k, curandState *rstates) {
  int jstart = ((long long)blockIdx.x) * nnz / gridDim.x;
  int jend = ((long long)(blockIdx.x + 1)) * nnz / gridDim.x;
  int id = threadIdx.x + k*blockIdx.x;
  curandState rstate; 
  if (threadIdx.x < k) {
    rstate = rstates[id];
  }
  for (int j = jstart; j < jend ; j++) {
    int aoff = nrows * Cir[j];
    int boff = nrows * Cic[j];
    float cr;
    if (threadIdx.x < k) {
      cr = P[j] * curand_uniform(&rstate);
    }
    int tid = threadIdx.x;
    float sum = 0;
    while (tid < nrows) {
      float tot = A[tid + aoff] * B[tid + boff];
      float tmp = __shfl_up(tot, 1);
      if (threadIdx.x >= 1) tot += tmp;
      tmp = __shfl_up(tot, 2);
      if (threadIdx.x >= 2) tot += tmp;
      tmp = __shfl_up(tot, 4);
      if (threadIdx.x >= 4) tot += tmp;
      tmp = __shfl_up(tot, 8);
      if (threadIdx.x >= 8) tot += tmp;
      tmp = __shfl_up(tot, 0x10);
      if (threadIdx.x >= 0x10) tot += tmp;

      float bsum = sum;
      sum += tot;
      tmp = __shfl_up(sum, 1);
      if (threadIdx.x > 0) {
        bsum = tmp;
      }
      for (int i = 0; i < k; i++) {
        float crx = __shfl(cr, i);
        if (crx > bsum && crx <= sum) {
          Ms[i + j*k] = tid + aoff;
          Us[i + j*k] = tid + boff;
        }
      }
      sum = __shfl(sum, 0x1f);
      tid += blockDim.x;
    }
          
  }
}


__global__ void __LDA_Gibbsy(int nrows, int ncols, float *A, float *B, float *AN, 
                             int *Cir, int *Cjc, float *P, float nsamps, curandState *rstates) {
  __shared__ float merge[32];
  int jstart = ((long long)blockIdx.x) * ncols / gridDim.x;
  int jend = ((long long)(blockIdx.x + 1)) * ncols / gridDim.x;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * blockIdx.x);
  curandState rstate = rstates[id];
  float prod, sum, bsum, user;
  int aoff, boff;
  for (int j0 = jstart; j0 < jend ; j0++) {
    boff = nrows * j0;
    user = B[tid + boff];
    for (int j = Cjc[j0]; j < Cjc[j0+1]; j++) {
      aoff = nrows * Cir[j];
      prod = A[tid + aoff] * user;
      sum = prod + __shfl_down(prod, 1);
      sum = sum + __shfl_down(sum, 2);
      sum = sum + __shfl_down(sum, 4);
      sum = sum + __shfl_down(sum, 8);
      sum = sum + __shfl_down(sum, 16);
      bsum = __shfl(sum, 0);
      __syncthreads();
      if (threadIdx.x == threadIdx.y) {
        merge[threadIdx.x] = bsum;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
        sum = merge[threadIdx.x];
        sum = sum + __shfl_down(sum, 1);
        sum = sum + __shfl_down(sum, 2);
        sum = sum + __shfl_down(sum, 4);
        sum = sum + __shfl_down(sum, 8);
        sum = sum + __shfl_down(sum, 16);
        bsum = __shfl(sum, 0);
        merge[threadIdx.x] = bsum;
      }
      __syncthreads();
      if (threadIdx.x == threadIdx.y) {
        sum = merge[threadIdx.x];
      }
      bsum = __shfl(sum, threadIdx.y);
      float pval = nsamps / bsum;
      int cr = curand_poisson(&rstate, prod * pval);
      if (cr > 0) {
        atomicAdd(&AN[tid + aoff], cr);
        user += cr;
      }
    }
    B[tid + boff] = user;
  }
}

//
// This version uses Poisson RNG to generate several random numbers per point, per iteration.
// nrows is number of rows in models A and B. A is nrows * nfeats, B is nrows * nusers
// AN and BN are updaters for A and B and hold sample counts from this iteration. 
// Cir anc Cic are row and column indices for the sparse matrix. 
// P holds the inner products (results of a call to dds) of A and B columns corresponding to Cir[j] and Cic[j]
// nsamps is the expected number of samples to compute for this iteration - its just a multiplier
// for individual poisson lambdas.
//
__global__ void __LDA_Gibbs(int nrows, int nnz, float *A, float *B, float *AN, float *BN, 
                            int *Cir, int *Cic, float *P, float nsamps, curandState *rstates) {
  int jstart = ((long long)blockIdx.x) * nnz / gridDim.x;
  int jend = ((long long)(blockIdx.x + 1)) * nnz / gridDim.x;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * blockIdx.x);
  curandState rstate = rstates[id];
  for (int j = jstart; j < jend ; j++) {
    int aoff = nrows * Cir[j];
    int boff = nrows * Cic[j];
    float pval = nsamps / P[j];
    for (int i = tid; i < nrows; i += blockDim.x * blockDim.y) {
      float prod = A[i + aoff] * B[i + boff];
      int cr = curand_poisson(&rstate, prod * pval);
      if (cr > 0) {
        atomicAdd(&AN[i + aoff], cr);
        atomicAdd(&BN[i + boff], cr);
      }
    }
  }
}

//
// Wait time Binomial generation for small np from Devroye's book:
// http://luc.devroye.org/rnbookindex.html chapter 10
// Time is directly proportional to the output value
//


__forceinline__ __device__ int __waittimegibbs(curandState *prstate, float p, int n) {
  if (p <= 0) return 0;
  float q = - log(1-p);
  float X = 0;
  float sum = 0;
  while (X < 100 && sum <= q) {
    float E = - log(curand_uniform(prstate));  // safe since curand_uniform wont return 0
    sum += E / (n - X);
    X += 1;
  }
  return X - 1;  
}

//
// Rejection Method for binomial random number generation from Devroye's book:
// http://luc.devroye.org/rnbookindex.html chapter 10
// Time is independent of output value, with a large constant
//

__forceinline__ __device__ int __binorndgibbs(float p, int n, curandState *prstate) {
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
    X = __waittimegibbs(prstate, p, n);      // Use a wait time method if small expected output
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
    X += __waittimegibbs(prstate, (oldp-p)/(1-p), n-X); // Now correct for rounding np to integer
  }
  if (pflipped) {                                       // correct for flipped p. 
    X = n - X;
  }
  return (int)X;
}


//
// This version uses Binomial RNG to generate several random numbers per point, per iteration.
// nrows is number of rows in models A and B. A is nrows * nfeats, B is nrows * nusers
// AN and BN are updaters for A and B and hold sample counts from this iteration. 
// Cir anc Cic are row and column indices for the sparse matrix. 
// P holds the inner products (results of a call to dds) of A and B columns corresponding to Cir[j] and Cic[j]
// nsamps is the total number of samples to compute for this iteration.
//

__global__ void __LDA_GibbsBino(int nrows, int nnz, float *A, float *B, float *AN, float *BN, 
                                int *Cir, int *Cic, float *Cv, float *P, int nsamps, curandState *rstates) {

  __shared__ int aoff[GBINOSIZE];
  __shared__ int boff[GBINOSIZE];
  __shared__ float mat[GBINOSIZE][GBINODIMX+1];

  int i, j, jrow, k, krow, valsleft, count;
  float vnorm, vv, cval;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  curandState *prstate = &rstates[tid + blockIdx.x*blockDim.x*blockDim.y];
  __syncthreads();
  for (i = GBINOSIZE*blockIdx.x; i < nnz; i += GBINOSIZE*gridDim.x) {     // Loop across blocks of GBINOSIZE nonzeros
    vnorm = 1.0f;
    cval = 0;
    if (i+tid < nnz) {                                        // Load the norms for these nonzeros
      cval = Cv[i+tid];
      aoff[tid] = nrows * Cir[i+tid];                         // load pointers to the columns to be multiplied
      boff[tid] = nrows * Cic[i+tid];
      vnorm = P[i+tid]*cval;
    }
    valsleft = (int)(cval*nsamps);                            // Initialize the count of samples for this column
    for (j = 0; j < nrows; j += blockDim.x) {                 // Loop over blocks of 32 rows
      __syncthreads();
      jrow = j+threadIdx.x;                                   // jrow is a global row index

      for (k = threadIdx.y; k < min(GBINOSIZE, nnz-i); k += blockDim.y) {    // Copy a 32x256 word block of product into SHMEM
        vv = 0;                                               // k is a local column index
        if (jrow < nrows) {
          vv = A[jrow + aoff[k]] * B[jrow + boff[k]];
        }
        mat[k][threadIdx.x] = vv;                             // get the block (effectively transposed)
      }
      __syncthreads();

      for (krow = 0; krow < min(blockDim.x, nrows-j); krow += 1) {    // Now walk down the columns with GBINOSIZE threads, krow is a local row index
        vv = min(mat[tid][krow], vnorm);                      // get the stored product value (guarded by vnorm)
        count = __binorndgibbs(vv/vnorm, valsleft, prstate);  // get a binomial random count
        count = min(count, valsleft);
        valsleft -= count;                                    // subtract it from remaining count
        vnorm -= vv;                                          // adjust remaining probability
        mat[tid][krow] = (float)count;                        // store count back in SHMEM
      }
      __syncthreads();

      for (k = threadIdx.y; k < min(GBINOSIZE, nnz-i); k += blockDim.y) {    // Add the 32x256 block of counts back into main memory, k indexes local columns
        vv = mat[k][threadIdx.x];
        if (jrow < nrows && vv > 0) {
          atomicAdd(&AN[jrow + aoff[k]], vv);
          atomicAdd(&BN[jrow + boff[k]], vv);
        }
      }
      __syncthreads();
    }
  }
} 

//
// This version includes a matrix nsamps of sample counts, one for each model coefficient. 
//

__global__ void __LDA_Gibbsv(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float *nsamps, curandState *rstates) {

  int jstart = ((long long)blockIdx.x) * nnz / gridDim.x;
  int jend = ((long long)(blockIdx.x + 1)) * nnz / gridDim.x;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * blockIdx.x);

  curandState rstate = rstates[id];

  for (int j = jstart; j < jend ; j++) {

    int aoff = nrows * Cir[j];
    int boff = nrows * Cic[j];

    float Pj = P[j];

    for (int i = tid; i < nrows; i += blockDim.x * blockDim.y) {

      float nsampsi = nsamps[i];
      float pval = nsampsi / Pj;
      float prod = A[i + aoff] * B[i + boff];

      int cr = curand_poisson(&rstate, prod * pval);
      if (cr > 0) {
        atomicAdd(&AN[i + aoff], cr/nsampsi );
        atomicAdd(&BN[i + boff], cr/nsampsi );
      }
    }
  }
}

__global__ void __randinit(curandState *rstates) {
  int id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * blockIdx.x);
  curand_init(1234, id, 0, &rstates[id]);
}
  

#else
__global__ void __LDA_Gibbs1(int nrows, int nnz, float *A, float *B, int *Cir, int *Cic, float *P, int *Ms, int *Us, int k, curandState *) {}
__global__ void __LDA_Gibbs(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float nsamps, curandState *) {}
__global__ void __LDA_Gibbsy(int nrows, int nnz, float *A, float *B, float *AN, int *Cir, int *Cic, float *P, float nsamps, curandState *) {}
__global__ void __LDA_Gibbsv(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float *nsamps, curandState *) {}
__global__ void __LDA_GibbsBino(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *Cv, float *P, int nsamps, curandState *rstates) {}
__global__ void __randinit(curandState *rstates) {}
#endif

#define DDS_BLKY 32

int LDA_Gibbs1(int nrows, int nnz, float *A, float *B, int *Cir, int *Cic, float *P, int *Ms, int *Us, int k) {
  int nblocks = min(1024, max(1,nnz/128));
  curandState *rstates;
  int err;
  err = cudaMalloc(( void **)& rstates , k * nblocks * sizeof(curandState));
  cudaDeviceSynchronize();
  __randinit<<<nblocks,k>>>(rstates); 
  cudaDeviceSynchronize();
  __LDA_Gibbs1<<<nblocks,32>>>(nrows, nnz, A, B, Cir, Cic, P, Ms, Us, k, rstates);
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}

int LDA_Gibbs(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float nsamps) {
  dim3 blockDims(32, min(32, 1+(nrows-1)/64), 1);
  int nblocks = min(128, max(1,nnz/128));
  curandState *rstates;
  int err;
  err = cudaMalloc(( void **)& rstates , nblocks * blockDims.x * blockDims.y * sizeof(curandState));
  cudaDeviceSynchronize();
  __randinit<<<nblocks,blockDims>>>(rstates); 
  cudaDeviceSynchronize(); 
  __LDA_Gibbs<<<nblocks,blockDims>>>(nrows, nnz, A, B, AN, BN, Cir, Cic, P, nsamps, rstates);
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}

int LDA_Gibbsy(int nrows, int ncols, float *A, float *B, float *AN, int *Cir, int *Cic, float *P, float nsamps) {
  dim3 blockDims(32, 32);
  int nblocks = min(128, max(1,ncols/2));
  curandState *rstates;
  int err;
  err = cudaMalloc(( void **)& rstates , nblocks * blockDims.x * blockDims.y * sizeof(curandState));
  cudaDeviceSynchronize();
  __randinit<<<nblocks,blockDims>>>(rstates); 
  cudaDeviceSynchronize(); 
  __LDA_Gibbsy<<<nblocks,blockDims>>>(nrows, ncols, A, B, AN, Cir, Cic, P, nsamps, rstates);
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}

int LDA_Gibbsv(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float *nsamps) {
  dim3 blockDims(min(32,nrows), min(32, 1+(nrows-1)/64), 1);
  int nblocks = min(128, max(1,nnz/128));
  curandState *rstates;
  int err;
  err = cudaMalloc(( void **)& rstates , nblocks * blockDims.x * blockDims.y * sizeof(curandState));
  cudaDeviceSynchronize();
  __randinit<<<nblocks,blockDims>>>(rstates); 
  cudaDeviceSynchronize();
  __LDA_Gibbsv<<<nblocks,blockDims>>>(nrows, nnz, A, B, AN, BN, Cir, Cic, P, nsamps, rstates);
  cudaDeviceSynchronize();
  cudaFree(rstates); 
  err = cudaGetLastError();
  return err;
}

int LDA_GibbsBino(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *Cv, float *P, int nsamps) {
  dim3 threads(GBINODIMX, GBINODIMY, 1);
  int nthreads = GBINODIMX*GBINODIMY;
  int nblocks = min(128, 1+ (nnz-1)/256);
  curandState *rstates;
  cudaError_t err = cudaMalloc(( void **)& rstates , nblocks * nthreads * sizeof(curandState));
  cudaDeviceSynchronize();
  __randinit<<<nblocks,threads>>>(rstates); 
  cudaDeviceSynchronize();
  __LDA_GibbsBino<<<nblocks,threads>>>(nrows, nnz, A, B, AN, BN, Cir, Cic, Cv, P, nsamps, rstates);
  cudaDeviceSynchronize();
  cudaFree(rstates);
  err = cudaGetLastError();
  return err;
}


