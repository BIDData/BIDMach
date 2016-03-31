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


__inline__ __device__ float inline_forward_sigmoid(float a) {
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

__inline__ __device__ float inline_deriv_sigmoid(float a, float d) {
  return d * (a - a * a);
}

__inline__ __device__ float inline_deriv_tanh(float a, float d) {
  return d * (1.0f - a * a);
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

__global__ void __lstm_fwd(float *inC, float *LIN1, float *LIN2, float *LIN3, float *LIN4, float *outC, float *outH, int n) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  float in_c, lin1, lin2, lin3, lin4;
  float in_gate, out_gate, forget_gate, in_sat, in_prod, f_prod, out_c, out_tanh, out_h;

  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    in_c = inC[i];
    lin1 = LIN1[i];
    lin2 = LIN2[i];
    lin3 = LIN3[i];
    lin4 = LIN4[i];

    in_gate = inline_forward_sigmoid(lin1);
    out_gate = inline_forward_sigmoid(lin2);
    forget_gate = inline_forward_sigmoid(lin3);
    in_sat = tanh(lin4);

    in_prod = in_gate * in_sat;
    f_prod = forget_gate * in_c;
    out_c = in_prod + f_prod;

    out_tanh = tanh(out_c);
    out_h = out_gate * out_tanh;

    outC[i] = out_c;
    outH[i] = out_h;  
  }
}


__global__ void __lstm_bwd(float *inC, float *LIN1, float *LIN2, float *LIN3, float *LIN4, float *doutC, float *doutH, 
                           float *dinC, float *dLIN1, float *dLIN2, float *dLIN3, float *dLIN4, int n) {
  int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  float in_c, lin1, lin2, lin3, lin4, in_gate, out_gate, forget_gate, in_sat, in_prod, f_prod, out_c, out_tanh;
  float din_c, dlin1, dlin2, dlin3, dlin4, din_gate, dout_gate, dforget_gate, din_sat, din_prod, df_prod, dout_c, dout_tanh, dout_h;

  for (int i = ip; i < n; i += blockDim.x * gridDim.x * gridDim.y) {
    in_c = inC[i];
    lin1 = LIN1[i];
    lin2 = LIN2[i];
    lin3 = LIN3[i];
    lin4 = LIN4[i];

    in_gate = inline_forward_sigmoid(lin1);
    out_gate = inline_forward_sigmoid(lin2);
    forget_gate = inline_forward_sigmoid(lin3);
    in_sat = tanh(lin4);

    in_prod = in_gate * in_sat;
    f_prod = forget_gate * in_c;
    out_c = in_prod + f_prod;

    out_tanh = tanh(out_c);
    //    out_h = out_gate * out_tanh;

    dout_h = doutH[i];
    dout_c = doutC[i];

    //    out_h = out_gate * out_tanh;
    dout_gate = dout_h * out_tanh;
    dout_tanh = dout_h * out_gate;

    //    out_tanh = tanh(out_c);
    dout_c += inline_deriv_tanh(out_c, dout_tanh);

    //    out_c = in_prod + f_prod;
    din_prod = dout_c;
    df_prod = dout_c;

    //    f_prod = forget_gate * in_c;
    dforget_gate = df_prod * in_c;
    din_c = df_prod * forget_gate;

    //    in_prod = in_gate * in_sat;
    din_gate = din_prod * in_sat;
    din_sat = din_prod * in_gate;

    //    in_gate = forward_sigmoid(lin1);
    //    out_gate = forward_sigmoid(lin2);
    //    forget_gate = forward_sigmoid(lin3);
    //    in_sat = tanh(lin4);

    dlin4 = inline_deriv_tanh(lin4, din_sat);
    dlin3 = inline_deriv_sigmoid(lin3, dforget_gate);
    dlin2 = inline_deriv_sigmoid(lin2, dout_gate);
    dlin1 = inline_deriv_sigmoid(lin1, din_gate);

    atomicAdd(&dLIN4[i], dlin4);
    atomicAdd(&dLIN3[i], dlin3);
    atomicAdd(&dLIN2[i], dlin2);
    atomicAdd(&dLIN1[i], dlin1);
    atomicAdd(&dinC[i], din_c);
  }
}


int lstm_fwd(float *inC, float *LIN1, float *LIN2, float *LIN3, float *LIN4, float *outC, float *outH, int n) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __lstm_fwd<<<griddims,nthreads>>>(inC, LIN1, LIN2, LIN3, LIN4, outC, outH, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

int lstm_bwd(float *inC, float *LIN1, float *LIN2, float *LIN3, float *LIN4, float *outC, float *outH, 
             float *dinC, float *dLIN1, float *dLIN2, float *dLIN3, float *dLIN4, int n) {
  int nthreads;
  dim3 griddims;
  setsizes(n, &griddims, &nthreads);
  __lstm_bwd<<<griddims,nthreads>>>(inC, LIN1, LIN2, LIN3, LIN4, outC, outH, dinC, dLIN1, dLIN2, dLIN3, dLIN4, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}

