#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <MatKernel.hpp>

typedef float (*fntype)(float);
typedef float (*optype)(float,float);

__device__ float forward_sigmoid(float a) {
  if (a > 20.0f) {
    return 1.0f;
  } else if (a < -80.0f) {
    return 0.0f;
  } else {
    return 1.0f/(1.0f + expf(-a));
  }
}

__device__ float forward_softplus(float a) {
  if (a > 20.0f) {
    return a;
  } else if (a < -20.0f) {
    return 0.0f;
  } else {
    return log1pf(expf(a));
  }
}

__device__ float forward_tanh(float a) {
  return tanhf(a);
}

__device__ float deriv_sigmoid(float a, float d) {
  return d * (a - a * a);
}

__device__ float deriv_tanh(float a, float d) {
  return d * (1.0f - a * a);
}

__device__ float deriv_softplus(float a, float d) {
  return d * forward_sigmoid(a);
}

__device__ const fntype forwardfns[] = {
  forward_sigmoid,
  tanh,
  forward_softplus
};

__device__ const optype derivfns[] = {
  deriv_sigmoid,
  deriv_tanh,
  deriv_softplus
};

__global__ void __apply_fwd(float *A, float *B, int ifn, int n) {
  fntype fn = forwardfns[ifn];
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    B[i] = fn(A[i]);
  }
}

int apply_fwd(float *A, float *B, int ifn, int n) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __apply_fwd<<<griddims,nthreads>>>(A, B, ifn, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void __apply_deriv(float *A, float *B, float *C, int ifn, int n) {
  optype fn = derivfns[ifn];
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    C[i] = fn(A[i], B[i]);
  }
}

int apply_deriv(float *A, float *B, float *C, int ifn, int n) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __apply_deriv<<<griddims,nthreads>>>(A, B, C, ifn, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}


