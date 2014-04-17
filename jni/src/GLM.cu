#include <cuda_runtime.h>
#include <stdio.h>
#include <MatKernel.hpp>

typedef float (*fntype)(float);
typedef float (*optype)(float,float);

__device__ float link_linear(float a) {return a;}
__device__ float link_logistic(float a) {return log(a/1.0f - a);}

__device__ float mean_linear(float a) {return a;}
__device__ float mean_logistic(float a) {
  float tmp;
  if (a > 0) {
    tmp = exp(-a);
    return 1.0f/(1.0f + tmp);
  } else {
    tmp = exp(a);
    return tmp/(1.0f + tmp);
  }
}

__device__ float deriv_linear(float a, float b) {return b-a;}
__device__ float deriv_logistic(float a, float b) {return b-a;}
__device__ float deriv_maxp(float p, float t) {return (2.0f*t - 1.0f)*p*(1.0f-p);}


#define eps 1.0e-10f
__device__ float ll_linear(float a, float t) {return (t-a)*(a-t);}
__device__ float ll_logistic(float a, float b) {return log(a * b + (1.0f - a) * (1.0f - b) + eps);}
__device__ float ll_maxp(float a, float t) {return a * t + (1.0f - a) * (1.0f - t) - 1.0f;}

__device__ const fntype linkfns[] = {
  link_linear,
  link_logistic,
  link_logistic};

__device__ const fntype meanfns[] = {
  mean_linear,
  mean_logistic,
  mean_logistic};

__device__ const optype derivfns[] = {
  deriv_linear,
  deriv_logistic,
  deriv_maxp};

__device__ const optype llfns[] = {
  ll_linear,
  ll_logistic,
  ll_maxp};


void setsizes(int N, dim3 *gridp, int *nthreadsp) {
  int nblocks = 1;
  int nthreads = 1;
  while (nblocks * nthreads < N) {
    if (nblocks < 16) {
      nblocks = 2*nblocks;
    } else if (nthreads < 1024) {
      nthreads = 2*nthreads;
    } else {
      nblocks = 2*nblocks;
    }
  }
  gridp->y = 1 + (nblocks-1)/65536;
  gridp->x = 1 + (nblocks-1)/gridp->y;
  gridp->z = 1;
  *nthreadsp = nthreads;
}

__global__ void __apply_preds(float *A, int *L, float *C, int nrows, int ncols) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < nrows*ncols; i += blockDim.x * gridDim.x * gridDim.y) {
    fntype fn = meanfns[L[i % nrows]];
    C[i] = fn(A[i]);
  }
}

int apply_preds(float *A, int *L, float *C, int nrows, int ncols) {
  int nthreads;
  dim3 griddims;
  setsizes(nrows*ncols, &griddims, &nthreads);
  __apply_preds<<<griddims,nthreads>>>(A, L, C, nrows, ncols);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void __apply_links(float *A, int *L, float *C, int nrows, int ncols) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < nrows*ncols; i += blockDim.x * gridDim.x * gridDim.y) {
    fntype fn = linkfns[L[i % nrows]];
    C[i] = fn(A[i]);
  }
}

int apply_links(float *A, int *L, float *C, int nrows, int ncols) {
  int nthreads;
  dim3 griddims;
  setsizes(nrows*ncols, &griddims, &nthreads);
  __apply_links<<<griddims,nthreads>>>(A, L, C, nrows, ncols);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void __apply_lls(float *A, float *B, int *L, float *C, int nrows, int ncols) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < nrows*ncols; i += blockDim.x * gridDim.x * gridDim.y) {
    optype op = llfns[L[i % nrows]];
    C[i] = op(A[i],B[i]);
  }
}


int apply_lls(float *A, float *B, int *L, float *C, int nrows, int ncols) {
  int nthreads;
  dim3 griddims;
  setsizes(nrows*ncols, &griddims, &nthreads);
  __apply_lls<<<griddims,nthreads>>>(A, B, L, C, nrows, ncols);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void __apply_derivs(float *A, float *B, int *L, float *C, int nrows, int ncols) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < nrows*ncols; i += blockDim.x * gridDim.x * gridDim.y) {
    optype op = derivfns[L[i % nrows]];
    C[i] = op(A[i],B[i]);
  }
}

int apply_derivs(float *A, float *B, int *L, float *C, int nrows, int ncols) {
  int nthreads;
  dim3 griddims;
  setsizes(nrows*ncols, &griddims, &nthreads);
  __apply_derivs<<<griddims,nthreads>>>(A, B, L, C, nrows, ncols);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}
