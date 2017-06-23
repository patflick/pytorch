#ifndef THP_CUDNN_DESCRIPTORS_INC
#define THP_CUDNN_DESCRIPTORS_INC

#include "Exceptions.h"

#include "cudnn2mio.h"

namespace torch { namespace cudnn {

struct TensorDescriptor
{
  cudnnTensorDescriptor_t desc;
  TensorDescriptor() : desc(NULL) {
    CHECK(cudnnCreateTensorDescriptor(&desc));
  }
  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor(TensorDescriptor&& ref)
  {
    desc = ref.desc;
    ref.desc = NULL;
  }
  ~TensorDescriptor() {
    cudnnDestroyTensorDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int dim, int* size, int* stride) {
    CHECK(cudnnSetTensorNdDescriptor(desc, dataType, dim, size, stride));
  }
};

struct FilterDescriptor
{
  cudnnFilterDescriptor_t desc;
  FilterDescriptor() : desc(NULL) {
    CHECK(cudnnCreateFilterDescriptor(&desc));
  }
  FilterDescriptor(const FilterDescriptor&) = delete;
  FilterDescriptor(FilterDescriptor&& ref)
  {
    desc = ref.desc;
    ref.desc = NULL;
  }
  ~FilterDescriptor() {
    cudnnDestroyFilterDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int* size) {
    CHECK(cudnnSetFilterNdDescriptor(desc, dataType, CUDNN_TENSOR_NCHW, 4, size));
  }
};

struct ConvolutionDescriptor
{
  cudnnConvolutionDescriptor_t desc;
  ConvolutionDescriptor() : desc(NULL) {
    CHECK(cudnnCreateConvolutionDescriptor(&desc));
  }
  ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
  ConvolutionDescriptor(ConvolutionDescriptor&& ref)
  {
    desc = ref.desc;
    ref.desc = NULL;
  }
  ~ConvolutionDescriptor() {
    cudnnDestroyConvolutionDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int* pad, int* stride) {
    int upscale[2] = {1, 1};
    CHECK(cudnnSetConvolutionNdDescriptor(desc, 2, pad, stride, upscale,
          CUDNN_CROSS_CORRELATION, dataType));
  }
};

}}  // namespace

#endif
