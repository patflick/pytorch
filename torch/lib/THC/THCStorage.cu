#include "THCStorage.h"

#ifdef THRUST_PATH
    #include <thrust/device_ptr.h>
    #include <thrust/fill.h>
    #if CUDA_VERSION >= 7000
        #include <thrust/system/cuda/execution_policy.h>
    #endif
#else
    #include <bolt/amp/fill.h>
    #include <bolt/amp/iterator/ubiquitous_iterator.h>
#endif

#include "THCHalf.h"

#include "generic/THCStorage.cu"
#include "THCGenerateAllTypes.h"
