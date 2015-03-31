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
// NOTE: The single-matrix version (hashmult) uses a fast lookup recurrence which is only valid up to 3000 base features per column (approx 4.5 million pairs)


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

const unsigned int seed = 3413413;

__forceinline__ __device__ unsigned int mmhashend(unsigned int hash, unsigned int mod)
{
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);
 
  return (hash % mod);
}

__forceinline__ __device__ unsigned int mmhash1(unsigned int v1, unsigned int mod) {
  unsigned int hash = seed;
  hash = h1(v1, hash);
  return mmhashend(hash, mod);
}
  
__forceinline__ __device__ unsigned int mmhash2(unsigned int v1, unsigned int v2, unsigned int mod) {
  unsigned int hash = seed;
  hash = h1(v1, hash);
  hash = h1(v2, hash);
  return mmhashend(hash, mod);
}

__forceinline__ __device__ unsigned int mmhash3(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int mod, unsigned int seed)
{
  unsigned int hash = seed;
 
  hash = h1(v1, hash);
  hash = h1(v2, hash);
  hash = h1(v3, hash);
  
  return mmhashend(hash, mod);
}
