#include "THCUNN.h"
#include "common.h"

struct tanhupdateOutput_functor
{
  __device__ void operator()(float *output, const float *input) const
  {
    *output = tanh(*input);
  }

  __device__ __host__
  ~tanhupdateOutput_functor() {}
};

void THNN_CudaTanh_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, tanhupdateOutput_functor());
}

struct tanhupdateGradInput_functor
{
  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = *gradOutput * (1 - *output * *output);
  }
 
  __device__ __host__
  ~tanhupdateGradInput_functor() {}
};

void THNN_CudaTanh_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, tanhupdateGradInput_functor());
}
