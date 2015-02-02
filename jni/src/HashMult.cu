#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

#if __CUDA_ARCH__ >= 300
#define MAXXGRID 2147483647
#else
#define MAXXGRID 65535
#endif



// Feature hashing multiply and multiply-transpose.
// This one enumerates, hashes and multiplies all pairs of features.
//
// NOTE: This uses a fast lookup recurrence which is only valid up to 3000 base features per column (approx 4.5 million pairs)


// Hash functions
// Adler32
__forceinline__ __device__ unsigned int adler32(const void *buf, size_t buflength) {
     const unsigned char *buffer = (const unsigned char*)buf;

     unsigned int s1 = 1;
     unsigned int s2 = 0;

     for (size_t n = 0; n < buflength; n++) {
        s1 = (s1 + buffer[n]) % 65521;
        s2 = (s2 + s1) % 65521;
     }     
     return (s2 << 16) | s1;
}

// MurmurHash3

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

__forceinline__ __device__ unsigned int mmhash2(unsigned int v1, unsigned int v2, unsigned int mod)
{
  unsigned int hash = 3413413;
 
  hash = h1(v1, hash);
  hash = h1(v2, hash);
  
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);
 
  return (hash % mod);
}

__forceinline__ __device__ int hash2(int a, int b, int modulus) {
  return mmhash2(a, b, modulus);
}

__forceinline__ __device__ int solve1(int j) {
  float v = sqrtf((float)j);
#pragma unroll
  for (int k = 0; k < 5; k++) {
    v = v - (v*(v+1)-2*j)/(2*v+1);   // Newton iterations to find first index. 
  }
  return (int)(v+2e-5f);   
}

__global__ void __hashmult(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *C) {
  int istart = ((long long)blockIdx.x) * ncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * ncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i];                                     // Range of nz rows in this column
    int jend = Bjc[i+1];
    int nr = jend - jstart;                                  // Number of nz rows
    int todo = nr * (nr + 1) / 2;                            // Number of pairs to process (including k,k pairs)
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = solve1(j);                                    // Compute the first and second indices
      int j2 = j - j1*(j1+1)/2; 
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1];                             // And their row indices
      int r2 = Bir[jstart + j2];
      int ind = hash2(r1, r2, nfeats);                       // Hash the indices

      float sum = A[threadIdx.x + nrows * ind] * f1 * f2;    // Do the product
      atomicAdd(&C[threadIdx.x + nrows * i], sum);
    }
  }
}

int hashmult(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *C) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, ncols);
  __hashmult<<<nblocks,threadDim>>>(nrows, nfeats, ncols, A, Bdata, Bir, Bjc, C);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void __hashmultT(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *C) {
  int istart = ((long long)blockIdx.x) * ncols/ gridDim.x;
  int iend = ((long long)(blockIdx.x + 1)) * ncols / gridDim.x;
  for (int i = istart; i < iend ; i++) {                     // i is the column index
    int jstart = Bjc[i];                                     // Range of nz rows in this column
    int jend = Bjc[i+1];
    int nr = jend - jstart;                                  // Number of nz rows
    int todo = nr * (nr + 1) / 2;                            // Number of pairs to process (including k,k pairs)
    for (int j = threadIdx.y; j < todo; j += blockDim.y) {   // j indexes a worker for this column
      int j1 = solve1(j);                                    // Compute the first and second indices
      int j2 = j - j1*(j1+1)/2; 
      float f1 = Bdata[jstart + j1];                         // Get the two features
      float f2 = Bdata[jstart + j2];
      int r1 = Bir[jstart + j1];                             // And their row indices
      int r2 = Bir[jstart + j2];
      int ind = hash2(r1, r2, nfeats);                       // Hash the indices

      float sum = A[threadIdx.x + nrows * i] * f1 * f2;      // Do the product
      atomicAdd(&C[threadIdx.x + nrows * ind], sum);
    }
  }
}

int hashmultT(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *C) {
  int nt = max(1, 256/nrows);
  dim3 threadDim(nrows, nt, 1);
  int nblocks = min(MAXXGRID, ncols);
  __hashmultT<<<nblocks,threadDim>>>(nrows, nfeats, ncols, A, Bdata, Bir, Bjc, C);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

