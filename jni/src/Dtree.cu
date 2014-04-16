#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

inline unsigned int h1(unsigned int k, unsigned int hash) {
  static const unsigned int c1 = 0xcc9e2d51;
  static const unsigned int c2 = 0x1b873593;
  static const unsigned int r1 = 15;
  static const unsigned int r2 = 13;
  static const unsigned int m = 5;
  static const unsigned int n = 0xe6546b64;

  k *= c1;
  k = (k << r1) | (k >> (32-r1));
  k *= c2;
 
  hash ^= k;
  hash = ((hash << r2) | (hash >> (32-r2)) * m) + n;
  return hash;
}

unsigned int mmhash(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int mod, unsigned int seed)
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
 
	return hash;
}


