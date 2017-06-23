/*
 * Wrapper of MiOpen in valid cudnn function API.
 * Allows to build code which uses cudnn, but compile with MiOpen instead.
 *
 * Currently supported:
 * - TensorDescriptor
 * - Convolutions (Fwd, BwdFilter, BwdData)
 */
#ifndef CUDNN2MIOPEN_H_GUARD
#define CUDNN2MIOPEN_H_GUARD

#ifndef WITH_MIOPEN
#include <cudnn.h>

#else // WITH_MIOPEN
// wrap MiOpen with cudnn calls
#include <miopen/miopen.h>

#define C2M_API_PRE static inline

#define CUDNNWINAPI static inline

// include during dev
//#include </opt/rocm/include/miopen/miopen.h>


/*
 * TODO:
 *    - IMPORTANT: map datatypes types (they are different!)
 *    - status mapping (mapping error codes to one another)
 */



typedef miopenStatus_t cudnnStatus_t; // TODO: match statuses
typedef miopenHandle_t cudnnHandle_t;
typedef miopenAcceleratorQueue_t cudaStream_t;

#define CUDNN_STATUS_SUCCESS miopenStatusSuccess

typedef hipError_t cudaError_t;

cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
    return miopenCreate(handle);
}
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
    return miopenDestroy(handle);
}
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t queue) {
    return miopenSetStream(handle, queue);
}
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *queue) {
    return miopenGetStream(handle, queue);
}

C2M_API_PRE const char* cudnnGetErrorString(cudnnStatus_t x) {
    return "Unkown Error (TODO: MiOpen error string)";
}

/*
 * CUDNN data types
 */
typedef enum
{
    CUDNN_DATA_HALF   = miopenHalf,
    CUDNN_DATA_FLOAT  = miopenFloat,
    CUDNN_DATA_DOUBLE = 20,
    /*
    CUDNN_DATA_INT8,
    CUDNN_DATA_INT32,
    CUDNN_DATA_INT8x4 = 5
    */
} cudnnDataType_t;

//typedef miopenDataType_t cudnnDataType_t; // TODO data type mapping between cuda types and miopen types (custom enum instead of typedef)



typedef miopenTensorDescriptor_t cudnnTensorDescriptor_t;

/* Create an instance of a generic Tensor descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(
                                        cudnnTensorDescriptor_t            *tensorDesc )
{
// Create a Tensor Descriptor
return miopenCreateTensorDescriptor(tensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[],
                                const int                           strideA[] ) {
// Not sure if the following two APIs are required right now
return miopenSetTensorDescriptor(
        tensorDesc,
        (miopenDataType_t)dataType,
        nbDims,
        (int*)dimA,
        (int*)strideA);
}

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t             tensorDesc ) {
    return miopenDestroyTensorDescriptor(tensorDesc);
}



typedef miopenTensorDescriptor_t cudnnFilterDescriptor_t;


/* Create an instance of FilterStruct: map to tensor descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(
                                cudnnFilterDescriptor_t            *filterDesc ) {
// Create a Tensor Descriptor
return miopenCreateTensorDescriptor(filterDesc);
}

typedef enum {
    // only supported one
    CUDNN_TENSOR_NCHW = 0,
} cudnnTensorFormat_t;

cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, // image data type
                                cudnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] ) {
    if (nbDims != 4)
        return miopenStatusNotImplemented; // TODO proper error
    return miopenSet4dTensorDescriptor(
	    filterDesc,
	    (miopenDataType_t)dataType,
            filterDimA[0],
	    filterDimA[1],
            filterDimA[2],
            filterDimA[3]);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(
                                cudnnFilterDescriptor_t             filterDesc ) {
    return miopenDestroyTensorDescriptor(filterDesc);
}


/*
typedef enum {
    miopenConvolution      = 0,
    miopenCrossCorrelation = 1,
} miopenConvolutionMode_t;
*/
//typedef int cudnnConvolutionMode_t
typedef enum {
	CUDNN_CONVOLUTION = miopenConvolution,
	CUDNN_CROSS_CORRELATION = miopenCrossCorrelation,
} cudnnConvolutionMode_t;

typedef miopenConvolutionDescriptor_t cudnnConvolutionDescriptor_t;

/* Create an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(
                                cudnnConvolutionDescriptor_t       *convDesc ) {
return miopenCreateConvolutionDescriptor(convDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc,
                                int                                 arrayLength,             /* nbDims-2 size */
                                const int                           padA[],
                                const int                           filterStrideA[],
                                const int                           dilationA[],
                                cudnnConvolutionMode_t              mode,
                                cudnnDataType_t                     computeType ) {
	if (arrayLength != 2) {
		// TODO proper error code?
		return miopenStatusBadParm;
	}
	return miopenInitConvolutionDescriptor(convDesc,
        (miopenConvolutionMode_t)(int)mode,
        padA[0],
        padA[1],
        filterStrideA[0],
        filterStrideA[1],
        dilationA[0],
        dilationA[1]);
}

/* Destroy an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc ) {
    return miopenDestroyConvolutionDescriptor(convDesc);
}


static inline miopenStatus_t myHipMallocTensorFromDesc(const miopenTensorDescriptor_t d, void** tensor) {
    // ASSERT 4D (only dim size supported by MiOpen)
    miopenDataType_t dataType;
    int n,c,h,w,ns,cs,hs,ws;
    miopenStatus_t stat = miopenGet4dTensorDescriptor(d, &dataType, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
    size_t size = n;
    size *= c; size *= h; size *= w;
    if (dataType == miopenHalf)
        size *= 2;
    if (dataType == miopenFloat)
        size *= 4;
    if (stat != 0)
        return stat;
    hipError_t err = hipMalloc(tensor, size); // XXX: wrong status type
    if (err != hipSuccess)
        return miopenStatusInternalError;
    return miopenStatusSuccess;
}


/* XXX: preferences are ignored by this MiOpen wrapper */
typedef enum
{
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;


/*
typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8,
} cudnnConvolutionFwdAlgo_t;
*/
typedef miopenConvFwdAlgorithm_t cudnnConvolutionFwdAlgo_t;

typedef int cudnnDeterminism_t; // XXX not needed right now

typedef struct {
    cudnnConvolutionFwdAlgo_t   algo;
    cudnnStatus_t               status;
    float                       time;
    size_t                      memory;
    cudnnDeterminism_t          determinism;
    int                         reserved[4];
} cudnnConvolutionFwdAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionFwdAlgoPerf_t      *perfResults ) {

    miopenConvAlgoPerf_t mio_perfs[requestedAlgoCount];

    /* allocate requried workspace */
    cudnnStatus_t stat;
    size_t workspace_size;
    stat = miopenConvolutionForwardGetWorkSpaceSize(
            handle,
            wDesc,
            xDesc,
            convDesc,
            yDesc,
            &workspace_size);

    if (stat != miopenStatusSuccess) {
        return stat;
    }

    void* workspace;
    hipError_t hiperr = hipMalloc(&workspace, workspace_size);
    if (hiperr != hipSuccess) {
        return miopenStatusInternalError; // XXX: wrong status type!
    }

    // allocate example tensors
    void *x, *y, *w;
    stat = myHipMallocTensorFromDesc(xDesc, &x);
    if (stat != 0)
        return stat;
    stat = myHipMallocTensorFromDesc(yDesc, &y);
    if (stat != 0)
        return stat;
    stat = myHipMallocTensorFromDesc(wDesc, &w);
    if (stat != 0)
        return stat;
    stat = miopenFindConvolutionForwardAlgorithm(handle,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        yDesc,
        y,
        requestedAlgoCount,
        returnedAlgoCount,
        mio_perfs,
        workspace,
        workspace_size,
        0);


    if (stat == miopenStatusSuccess) {
            for (int i = 0; i < *returnedAlgoCount; ++i) {
                perfResults[i].time =   mio_perfs[i].time;
                perfResults[i].memory = mio_perfs[i].memory;
                perfResults[i].algo =   mio_perfs[i].fwd_algo;
            }
    }

    return stat;
}


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdPreference_t     preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionFwdAlgo_t          *algo ) {
	*algo = miopenConvolutionFwdAlgoGEMM; // TODO: actual selection
	return miopenStatusSuccess;
};

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                size_t                             *sizeInBytes ) {
return miopenConvolutionForwardGetWorkSpaceSize(
        handle,
        wDesc,
        xDesc,
        convDesc,
        yDesc,
        sizeInBytes);
}

/* Function to perform the forward pass for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y ) {

return miopenConvolutionForward(handle,
        alpha,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        algo, /* XXX: algo mapping? */
        beta,
        yDesc,
        y,
        workSpace,
        workSpaceSizeInBytes);
}



// XXX: algorithm preference is currently ignored by this MiOpen wrapper
typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

/*
typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                 = 0,  // non-deterministic
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                 = 3,  // non-deterministic, algo0 with workspace
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD          = 4,  // not implemented
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING        = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT             = 7,   
} cudnnConvolutionBwdFilterAlgo_t;
*/
typedef miopenConvBwdWeightsAlgorithm_t cudnnConvolutionBwdFilterAlgo_t;


// torch only reads the `algo` field of the `...perf_t` types

/*
// Same perf struct for forward, backward filter and backward data algorthms
typedef struct{
    union {
        miopenConvFwdAlgorithm_t fwd_algo;
        miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
        miopenConvBwdDataAlgorithm_t bwd_data_algo;
    };
    float time;
    size_t memory;
} miopenConvAlgoPerf_t;
*/


typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t                   status;
    float                           time;
    size_t                          memory;
    cudnnDeterminism_t              determinism;
    int                             reserved[4];
} cudnnConvolutionBwdFilterAlgoPerf_t;


cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       dwDesc,
                                const int                           requestedAlgoCount,
                                int                                 *returnedAlgoCount,
                                cudnnConvolutionBwdFilterAlgoPerf_t *perfResults ) {

    miopenConvAlgoPerf_t mio_perfs[requestedAlgoCount];

    /* allocate requried workspace */
    cudnnStatus_t stat;
    size_t workspace_size;
    stat = miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle,
            dyDesc,
            xDesc,
            convDesc,
            dwDesc,
            &workspace_size);

    if (stat != miopenStatusSuccess) {
        return stat;
    }

    void* workspace;
    hipError_t hiperr = hipMalloc(&workspace, workspace_size);
    if (hiperr != hipSuccess) {
        return miopenStatusInternalError; // XXX: wrong status type!
    }

    // allocate example tensors
    void *x, *dy, *dw;
    stat = myHipMallocTensorFromDesc(xDesc, &x);
    if (stat != 0)
        return stat;
    stat = myHipMallocTensorFromDesc(dyDesc, &dy);
    if (stat != 0)
        return stat;
    stat = myHipMallocTensorFromDesc(dwDesc, &dw);
    if (stat != 0)
        return stat;


    stat = miopenFindConvolutionBackwardWeightsAlgorithm(handle,
            dyDesc,
            dy,
            xDesc,
            x,
            convDesc,
            dwDesc,
            dw,
            requestedAlgoCount,
            returnedAlgoCount,
            mio_perfs,
            workspace,
            workspace_size,
            0);

    if (stat == miopenStatusSuccess) {
            for (int i = 0; i < *returnedAlgoCount; ++i) {
                perfResults[i].time =   mio_perfs[i].time;
                perfResults[i].memory = mio_perfs[i].memory;
                perfResults[i].algo =   mio_perfs[i].bwd_weights_algo;
            }
    }

    return stat;
}


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
                                cudnnHandle_t                         handle,
                                const cudnnTensorDescriptor_t         xDesc,
                                const cudnnTensorDescriptor_t         dyDesc,
                                const cudnnConvolutionDescriptor_t    convDesc,
                                const cudnnFilterDescriptor_t         dwDesc,
                                cudnnConvolutionBwdFilterPreference_t preference,
                                size_t                                memoryLimitInBytes,
                                cudnnConvolutionBwdFilterAlgo_t      *algo ) {
	// TODO: actual algo selection
	*algo = miopenConvolutionBwdWeightsAlgoGEMM;
	return miopenStatusSuccess;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       gradDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                size_t                             *sizeInBytes ) {
return miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        handle,
        dyDesc,
        xDesc,
        convDesc,
        gradDesc,
        sizeInBytes);
}

typedef miopenConvBwdWeightsAlgorithm_t cudnnConvolutionBwdFilterAlgo_t;

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnFilterDescriptor_t       dwDesc,
                                void                               *dw ) {
    return miopenConvolutionBackwardWeights(handle,
	    alpha,
	    dyDesc,
	    dy,
	    xDesc,
	    x,
	    convDesc,
	    algo,
	    beta,
	    dwDesc,
	    dw,
	    workSpace,
	    workSpaceSizeInBytes);
}



// algo preferences are ignored by this MiOpen wrapper
typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE             = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST           = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = 2,
} cudnnConvolutionBwdDataPreference_t;

/*
typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0, // non-deterministic
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT             = 6,
} cudnnConvolutionBwdDataAlgo_t;
*/

typedef miopenConvBwdDataAlgorithm_t cudnnConvolutionBwdDataAlgo_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t   algo;
    cudnnStatus_t                   status;
    float                           time;
    size_t                          memory;
    cudnnDeterminism_t              determinism;
    int                             reserved[4];
} cudnnConvolutionBwdDataAlgoPerf_t;


cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionBwdDataAlgoPerf_t  *perfResults ) {

    miopenConvAlgoPerf_t mio_perfs[requestedAlgoCount];

    /* allocate requried workspace */
    cudnnStatus_t stat;
    size_t workspace_size;
    stat = miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle,
            dyDesc,
            wDesc,
            convDesc,
            dxDesc,
            &workspace_size);

    if (stat != miopenStatusSuccess) {
        return stat;
    }

    void* workspace;
    hipError_t hiperr = hipMalloc(&workspace, workspace_size);
    if (hiperr != hipSuccess) {
        return miopenStatusInternalError; // XXX: wrong status type!
    }

    // allocate example tensors
    void *dx, *dy, *w;
    stat = myHipMallocTensorFromDesc(dxDesc, &dx);
    if (stat != 0)
        return stat;
    stat = myHipMallocTensorFromDesc(dyDesc, &dy);
    if (stat != 0)
        return stat;
    stat = myHipMallocTensorFromDesc(wDesc, &w);
    if (stat != 0)
        return stat;

    stat = miopenFindConvolutionBackwardDataAlgorithm(handle,
            dyDesc,
            dy,
            wDesc,
            w,
            convDesc,
            dxDesc,
            dx,
            requestedAlgoCount,
            returnedAlgoCount,
            mio_perfs,
            workspace,
            workspace_size,
            0);

    if (stat == miopenStatusSuccess) {
            for (int i = 0; i < *returnedAlgoCount; ++i) {
                perfResults[i].time =   mio_perfs[i].time;
                perfResults[i].memory = mio_perfs[i].memory;
                perfResults[i].algo =   mio_perfs[i].bwd_data_algo;
            }
    }

    return stat;
}


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataPreference_t preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionBwdDataAlgo_t      *algo ) {
	*algo = miopenConvolutionBwdDataAlgoGEMM;
        return miopenStatusSuccess;
}


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                size_t                             *sizeInBytes ) {
return miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle,
        dyDesc,
        wDesc,
        convDesc,
        dxDesc,
        sizeInBytes);
}



cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx ) {
return miopenConvolutionBackwardData(handle,
        alpha,
        dyDesc,
        dy,
        wDesc,
        w,
        convDesc,
        algo, // TODO: algo mapping?
        beta,
        dxDesc,
        dx,
        workSpace,
        workSpaceSizeInBytes);
}



/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C ){
// map Y = alpha * B + beta * Y (MiOpen Beta: alpha = 1, beta = 0, always???? TODO FIXME)
return miopenConvolutionForwardBias(handle,
        alpha,
        aDesc,
        A,
        beta,
        cDesc,
        C);
}

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dbDesc,
                                void                               *db ) {
return miopenConvolutionBackwardBias(handle,
        alpha,
        dyDesc,
        dy,
        beta,
        dbDesc,
        db);
}

#endif // WITH_MIOPEN

#endif // CUDNN2MIOPEN_H_GUARD
