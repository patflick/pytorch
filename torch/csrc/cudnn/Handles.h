#ifndef THP_CUDNN_HANDLE_INC
#define THP_CUDNN_HANDLE_INC

#include "cudnn2mio.h"

namespace torch { namespace cudnn {

cudnnHandle_t getCudnnHandle();

}} // namespace

#endif
