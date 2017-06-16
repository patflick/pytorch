#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCTensorMathReduce.cuh"
#include "THCTensorMathPointwise.cuh"

#ifdef THRUST_PATH
  #include <thrust/device_ptr.h>
  #include <thrust/transform_reduce.h>
  #include <thrust/functional.h>
  #include <thrust/inner_product.h>
  #if CUDA_VERSION >= 7000
    #include <thrust/system/cuda/execution_policy.h>
  #endif
#else
  #include <bolt/amp/functional.h>
  #include <bolt/amp/inner_product.h>
  #include <bolt/amp/iterator/ubiquitous_iterator.h>
#endif

struct TensorTPowOp {
  __host__ __device__
  explicit
  TensorTPowOp(float v) : val(v) {}

  __device__ __forceinline__
  void operator()(float* out, float* in) { *out = powf(val, *in); }

  __device__ __forceinline__
  void operator()(float* v) { *v = powf(val, *v); }

  float val;
};

void THCudaTensor_tpow(
    THCState *state, THCudaTensor *self_, float value, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);
    if (!THC_pointwiseApply2(state, self_, src, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(hipGetLastError());
}

struct TensorATan2Op {
  __device__ __forceinline__
  void operator()(float* out, float* a, float* b) { *out = atan2f(*a, *b); }
};

void THCudaTensor_atan2(
    THCState *state, THCudaTensor *self_, THCudaTensor *tx, THCudaTensor *ty)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, tx, ty));
  THArgCheck(THCudaTensor_nElement(state, tx) ==
             THCudaTensor_nElement(state, ty), 3, "sizes do not match");
  THCudaTensor_resizeAs(state, self_, tx);
  if (!THC_pointwiseApply3(state, self_, tx, ty, TensorATan2Op())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(hipGetLastError());
}

float THCudaTensor_dist(
    THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  self = THCudaTensor_newContiguous(state, self);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  src = THCudaTensor_newContiguous(state, src);
  float result = 0;
#ifdef THRUST_PATH
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(state, src));

  result = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+size, src_data, (float) 0,
    thrust::plus<float>(), TensorDistOp<float>(value));

#else
    auto self_data =
        bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, self));
    auto src_data =
        bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, src));

    result = bolt::amp::inner_product(
        self_data,
        self_data + size,
        src_data,
        0.0f,
        bolt::amp::plus<float>(),
        TensorDistOp<float>(value));
#endif

  THCudaTensor_free(state, src);
  THCudaTensor_free(state, self);

  return pow(result, (float)1.0/value);
}

void THCudaTensor_rand(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_uniform(state, r_, 0, 1);
}

void THCudaTensor_randn(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_normal(state, r_, 0, 1);
}
