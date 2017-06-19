#include <Python.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <stdbool.h>
#include <unordered_map>
#include <libshm.h>
#include <TH/TH.h>

#ifdef WITH_CUDNN
#include "cudnn/Module.h"
#endif

#define WITH_NUMPY_IMPORT_ARRAY
#include "THP.h"

PyObject* module;
PyObject* tensor_classes;

PyObject *THPDefaultTensorClass = NULL;
THPGenerator *THPDefaultGenerator   = NULL;

static bool THPModule_loadClasses(PyObject *self)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  PyObject *torch_module = PyImport_ImportModule("torch");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return false;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);

  ASSERT_NOT_NULL(tensor_classes = PyMapping_GetItemString(module_dict, (char*)"_tensor_classes"));

  ASSERT_NOT_NULL(THPDoubleStorageClass = PyMapping_GetItemString(module_dict,(char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THPFloatStorageClass  = PyMapping_GetItemString(module_dict,(char*)"FloatStorage"));
  ASSERT_NOT_NULL(THPLongStorageClass   = PyMapping_GetItemString(module_dict,(char*)"LongStorage"));
  ASSERT_NOT_NULL(THPIntStorageClass    = PyMapping_GetItemString(module_dict,(char*)"IntStorage"));
  ASSERT_NOT_NULL(THPShortStorageClass  = PyMapping_GetItemString(module_dict,(char*)"ShortStorage"));
  ASSERT_NOT_NULL(THPCharStorageClass   = PyMapping_GetItemString(module_dict,(char*)"CharStorage"));
  ASSERT_NOT_NULL(THPByteStorageClass   = PyMapping_GetItemString(module_dict,(char*)"ByteStorage"));

  ASSERT_NOT_NULL(THPDoubleTensorClass  = PyMapping_GetItemString(module_dict,(char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THPFloatTensorClass   = PyMapping_GetItemString(module_dict,(char*)"FloatTensor"));
  ASSERT_NOT_NULL(THPLongTensorClass    = PyMapping_GetItemString(module_dict,(char*)"LongTensor"));
  ASSERT_NOT_NULL(THPIntTensorClass     = PyMapping_GetItemString(module_dict,(char*)"IntTensor"));
  ASSERT_NOT_NULL(THPShortTensorClass   = PyMapping_GetItemString(module_dict,(char*)"ShortTensor"));
  ASSERT_NOT_NULL(THPCharTensorClass    = PyMapping_GetItemString(module_dict,(char*)"CharTensor"));
  ASSERT_NOT_NULL(THPByteTensorClass    = PyMapping_GetItemString(module_dict,(char*)"ByteTensor"));

  return true;
#undef ASSERT_NOT_NULL
}

////////////////////////////////////////////////////////////////////////////////
// Copy handlers
////////////////////////////////////////////////////////////////////////////////

#include "ModuleCopy.h"

std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> tensor_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> storage_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> tensor_async_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> storage_async_copy_handlers;

#define COPY_METHODS(name) TH_CONCAT_2(name,_copy_handlers)
#define IMPLEMENT_COPY_WITH_WRAPPER(name)                                      \
bool TH_CONCAT_3(THPModule_,name,Copy)(PyObject *dst, PyObject *src)           \
{                                                                              \
  auto it = COPY_METHODS(name).find(std::make_pair((PyObject*)Py_TYPE(dst), (PyObject*)Py_TYPE(src))); \
  if (it == COPY_METHODS(name).end()) {                                        \
    THPUtils_setError("Copy function from %s to %s isn't implemented!", Py_TYPE(src)->tp_name, Py_TYPE(dst)->tp_name); \
    return false;                                                              \
  }                                                                            \
  (it->second)(dst, src);                                                      \
  return true;                                                                 \
}                                                                              \
                                                                               \
static PyObject * TH_CONCAT_3(THPModule_,name,CopyWrapper)(PyObject *unused, PyObject *args)\
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;                         \
  THPUtils_assert(num_args == 2, #name "Copy expected exactly two arguments, " \
          "but got %ld", (long)num_args);                                      \
  PyObject *dst = PyTuple_GET_ITEM(args, 0);                                   \
  PyObject *src = PyTuple_GET_ITEM(args, 1);                                   \
  if (!TH_CONCAT_3(THPModule_,name,Copy)(dst, src)) {                          \
    return NULL;                                                               \
  }                                                                            \
  Py_INCREF(dst);                                                              \
  return dst;                                                                  \
  END_HANDLE_TH_ERRORS                                                         \
}

IMPLEMENT_COPY_WITH_WRAPPER(tensor)
IMPLEMENT_COPY_WITH_WRAPPER(storage)
IMPLEMENT_COPY_WITH_WRAPPER(tensor_async)
IMPLEMENT_COPY_WITH_WRAPPER(storage_async)
#undef COPY_METHODS
#undef IMPLEMENT_COPY_WITH_WRAPPER

#include "ModuleCopy.cpp"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static bool THPModule_assignStateless(PyObject *self)
{
#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_2(type, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THP,type,TensorClass), THP_STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS(Float);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *self, PyObject *shm_manager_path)
{
  if (!THPUtils_checkBytes(shm_manager_path)) {
    THPUtils_setError("initialization error - expected bytes/string object as shm_manager_path!");
    return NULL;
  }
  libshm_init(THPUtils_bytesAsString(shm_manager_path));
  if (!THPModule_loadClasses(self))         return NULL;
  if (!THPModule_assignStateless(self))     return NULL;
  if (!THPModule_initCopy(self))            return NULL;
  return PyBool_FromLong(true);
}

static PyObject * THPModule_getNumThreads(PyObject *module)
{
  return PyLong_FromLong(THGetNumThreads());
}

static PyObject * THPModule_setNumThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
  THSetNumThreads((int)THPUtils_unpackLong(arg));
  Py_RETURN_NONE;
}

bool THPModule_isTensor(PyObject *obj)
{
  int result = PySet_Contains(tensor_classes, (PyObject*)Py_TYPE(obj));
  if (result == -1)
    throw std::logic_error("FATAL: tensor_classes isn't a set!");
  return result;
}

// Following two functions are taken from Python 2.7 source
// https://github.com/python/cpython/blob/2.7/Modules/_multiprocessing/multiprocessing.c
static PyObject * THPModule_sendfd(PyObject *self, PyObject *args)
{
  int conn, fd, res;
  struct iovec dummy_iov;
  char dummy_char;
  struct msghdr msg;
  struct cmsghdr *cmsg;
  union {
    struct cmsghdr hdr;
    unsigned char buf[CMSG_SPACE(sizeof(int))];
  } cmsgbuf;

  if (!PyArg_ParseTuple(args, "ii", &conn, &fd))
    return NULL;

  dummy_iov.iov_base = &dummy_char;
  dummy_iov.iov_len = 1;

  memset(&msg, 0, sizeof(msg));
  msg.msg_control = &cmsgbuf.buf;
  msg.msg_controllen = sizeof(cmsgbuf.buf);
  msg.msg_iov = &dummy_iov;
  msg.msg_iovlen = 1;

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  * (int *) CMSG_DATA(cmsg) = fd;

  Py_BEGIN_ALLOW_THREADS
  res = sendmsg(conn, &msg, 0);
  Py_END_ALLOW_THREADS

  if (res < 0)
    return PyErr_SetFromErrno(PyExc_OSError);
  Py_RETURN_NONE;
}

static PyObject * THPModule_recvfd(PyObject *self, PyObject *args)
{
  int conn, fd, res;
  char dummy_char;
  struct iovec dummy_iov;
  struct msghdr msg = {0};
  struct cmsghdr *cmsg;
  union {
    struct cmsghdr hdr;
    unsigned char buf[CMSG_SPACE(sizeof(int))];
  } cmsgbuf;

  if (!PyArg_ParseTuple(args, "i", &conn))
    return NULL;

  dummy_iov.iov_base = &dummy_char;
  dummy_iov.iov_len = 1;

  memset(&msg, 0, sizeof(msg));
  msg.msg_control = &cmsgbuf.buf;
  msg.msg_controllen = sizeof(cmsgbuf.buf);
  msg.msg_iov = &dummy_iov;
  msg.msg_iovlen = 1;

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  msg.msg_controllen = cmsg->cmsg_len;

  Py_BEGIN_ALLOW_THREADS
  res = recvmsg(conn, &msg, 0);
  Py_END_ALLOW_THREADS

  if (res < 0)
    return PyErr_SetFromErrno(PyExc_OSError);

  if (msg.msg_controllen < CMSG_LEN(sizeof(int)) ||
      (cmsg = CMSG_FIRSTHDR(&msg)) == NULL ||
      cmsg->cmsg_level != SOL_SOCKET ||
      cmsg->cmsg_type != SCM_RIGHTS ||
      cmsg->cmsg_len < CMSG_LEN(sizeof(int))) {
    /* If at least one control message is present, there should be
        no room for any further data in the buffer. */
    PyErr_SetString(PyExc_RuntimeError, "No file descriptor received");
    return NULL;
  }

  fd = * (int *) CMSG_DATA(cmsg);
  return Py_BuildValue("i", fd);
}

PyObject * THPModule_setDefaultTensorType(PyObject *_unused, PyObject *type)
{
  THPDefaultTensorClass = type;
  Py_RETURN_NONE;
}

PyObject * THPModule_fromNumpy(PyObject *_unused, PyObject *array)
{
#ifndef WITH_NUMPY
  THPUtils_setError("torch was compiled without numpy support");
  return NULL;
#else
  THPUtils_assert(PyArray_Check(array), "from_numpy expects an np.ndarray "
      "but got %s", THPUtils_typename(array));
  int type = PyArray_TYPE((PyArrayObject*)array);
  if (type == NPY_DOUBLE) {
    return PyObject_CallFunctionObjArgs(THPDoubleTensorClass, array, NULL);
  } else if (type == NPY_FLOAT) {
    return PyObject_CallFunctionObjArgs(THPFloatTensorClass, array, NULL);
  } else if (type == NPY_INT64) {
    return PyObject_CallFunctionObjArgs(THPLongTensorClass, array, NULL);
  } else if (type == NPY_INT32) {
    return PyObject_CallFunctionObjArgs(THPIntTensorClass, array, NULL);
  } else if (type == NPY_UINT8) {
    return PyObject_CallFunctionObjArgs(THPByteTensorClass, array, NULL);
  }
  THPUtils_setError("can't convert a given np.ndarray to a tensor - it has an "
      "invalid type. The only supported types are: double, float, int64, "
      "int32, and uint8.");
  return NULL;
#endif
}


#define IMPLEMENT_STATELESS(name)                                              \
static PyObject * TH_CONCAT_2(THPModule_, name)(PyObject *_unused, PyObject *args, PyObject *kwargs) \
{                                                                              \
  PyObject *tensor = THPDefaultTensorClass;                                    \
  PyObject *key, *value;                                                       \
  Py_ssize_t pos = 0;                                                          \
  for (int i = 0; i < PyTuple_Size(args); i++) {                               \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item)) {                                            \
      tensor = item;                                                           \
      goto dispatch;                                                           \
    }                                                                          \
  }                                                                            \
  if (kwargs) {                                                                \
    while (PyDict_Next(kwargs, &pos, &key, &value)) {                          \
      if (THPModule_isTensor(value)) {                                         \
        tensor = value;                                                        \
        goto dispatch;                                                         \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
dispatch:                                                                      \
  THPObjectPtr methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME); \
  THPUtils_assert(methods, "Type %s doesn't implement stateless methods",      \
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor)); \
  THPObjectPtr method = PyObject_GetAttrString(methods, #name);                \
  THPUtils_assert(method, "Type %s doesn't implement stateless method " #name, \
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor)); \
  return PyObject_Call(method, args, kwargs);                                  \
}

IMPLEMENT_STATELESS(sigmoid)
IMPLEMENT_STATELESS(log)
IMPLEMENT_STATELESS(log1p)
IMPLEMENT_STATELESS(exp)
IMPLEMENT_STATELESS(cos)
IMPLEMENT_STATELESS(acos)
IMPLEMENT_STATELESS(cosh)
IMPLEMENT_STATELESS(sin)
IMPLEMENT_STATELESS(asin)
IMPLEMENT_STATELESS(sinh)
IMPLEMENT_STATELESS(tan)
IMPLEMENT_STATELESS(atan)
IMPLEMENT_STATELESS(tanh)
IMPLEMENT_STATELESS(sqrt)
IMPLEMENT_STATELESS(rsqrt)
IMPLEMENT_STATELESS(ceil)
IMPLEMENT_STATELESS(floor)
IMPLEMENT_STATELESS(round)
IMPLEMENT_STATELESS(abs)
IMPLEMENT_STATELESS(trunc)
IMPLEMENT_STATELESS(frac)
IMPLEMENT_STATELESS(mean)
IMPLEMENT_STATELESS(std)
IMPLEMENT_STATELESS(var)
IMPLEMENT_STATELESS(norm)
IMPLEMENT_STATELESS(cinv)
IMPLEMENT_STATELESS(neg)
IMPLEMENT_STATELESS(add)
IMPLEMENT_STATELESS(csub)
IMPLEMENT_STATELESS(mul)
IMPLEMENT_STATELESS(div)
IMPLEMENT_STATELESS(fmod)
IMPLEMENT_STATELESS(cmul)
IMPLEMENT_STATELESS(cdiv)
IMPLEMENT_STATELESS(cfmod)
IMPLEMENT_STATELESS(min)
IMPLEMENT_STATELESS(max)
IMPLEMENT_STATELESS(cmax)
IMPLEMENT_STATELESS(cmin)
IMPLEMENT_STATELESS(cpow)
IMPLEMENT_STATELESS(dot)
IMPLEMENT_STATELESS(sum)
IMPLEMENT_STATELESS(prod)
IMPLEMENT_STATELESS(remainder)
IMPLEMENT_STATELESS(cremainder)
IMPLEMENT_STATELESS(cumsum)
IMPLEMENT_STATELESS(cumprod)
IMPLEMENT_STATELESS(clamp)
IMPLEMENT_STATELESS(equal)
IMPLEMENT_STATELESS(eye)
IMPLEMENT_STATELESS(fill)
IMPLEMENT_STATELESS(diag)
IMPLEMENT_STATELESS(numel)
IMPLEMENT_STATELESS(sign)
IMPLEMENT_STATELESS(trace)
IMPLEMENT_STATELESS(tril)
IMPLEMENT_STATELESS(triu)
IMPLEMENT_STATELESS(zero)
IMPLEMENT_STATELESS(kthvalue)
IMPLEMENT_STATELESS(mode)
IMPLEMENT_STATELESS(median)
IMPLEMENT_STATELESS(cross)
//IMPLEMENT_STATELESS(sort)
//IMPLEMENT_STATELESS(topk)
IMPLEMENT_STATELESS(t)
IMPLEMENT_STATELESS(transpose)
IMPLEMENT_STATELESS(squeeze)
IMPLEMENT_STATELESS(renorm)
IMPLEMENT_STATELESS(dist)
IMPLEMENT_STATELESS(linspace)
IMPLEMENT_STATELESS(logspace)
IMPLEMENT_STATELESS(histc)
IMPLEMENT_STATELESS(atan2)
IMPLEMENT_STATELESS(pow)
IMPLEMENT_STATELESS(lerp)
IMPLEMENT_STATELESS(reshape)
IMPLEMENT_STATELESS(zeros)
IMPLEMENT_STATELESS(ones)
IMPLEMENT_STATELESS(index_select)
IMPLEMENT_STATELESS(narrow)
IMPLEMENT_STATELESS(addmm)
IMPLEMENT_STATELESS(addmv)
IMPLEMENT_STATELESS(addr)
IMPLEMENT_STATELESS(ger)
IMPLEMENT_STATELESS(mv)
IMPLEMENT_STATELESS(addbmm)
IMPLEMENT_STATELESS(baddbmm)
IMPLEMENT_STATELESS(addcmul)
IMPLEMENT_STATELESS(addcdiv)
IMPLEMENT_STATELESS(mm)
IMPLEMENT_STATELESS(bmm)
// TODO: this doesn't implement options that return numbers!
IMPLEMENT_STATELESS(multinomial)
IMPLEMENT_STATELESS(uniform)
IMPLEMENT_STATELESS(normal)
IMPLEMENT_STATELESS(cauchy)
IMPLEMENT_STATELESS(log_normal)
IMPLEMENT_STATELESS(exponential)
IMPLEMENT_STATELESS(random)
IMPLEMENT_STATELESS(geometric)
IMPLEMENT_STATELESS(bernoulli)
IMPLEMENT_STATELESS(randperm)
IMPLEMENT_STATELESS(unfold)
IMPLEMENT_STATELESS(range)
IMPLEMENT_STATELESS(gather)
IMPLEMENT_STATELESS(scatter)
IMPLEMENT_STATELESS(rand)
IMPLEMENT_STATELESS(randn)
IMPLEMENT_STATELESS(all)
IMPLEMENT_STATELESS(any)
IMPLEMENT_STATELESS(masked_select)
IMPLEMENT_STATELESS(gesv)
IMPLEMENT_STATELESS(gels)
IMPLEMENT_STATELESS(trtrs)
IMPLEMENT_STATELESS(symeig)
IMPLEMENT_STATELESS(eig)
IMPLEMENT_STATELESS(svd)
IMPLEMENT_STATELESS(inverse)
IMPLEMENT_STATELESS(potrf)
IMPLEMENT_STATELESS(potrs)
IMPLEMENT_STATELESS(potri)
IMPLEMENT_STATELESS(pstrf)
IMPLEMENT_STATELESS(qr)
IMPLEMENT_STATELESS(geqrf)
IMPLEMENT_STATELESS(orgqr)
IMPLEMENT_STATELESS(ormqr)

#undef IMPLEMENT_STATELESS

// For logical functions a reverse type search is required (if the first argument
// is a ByteTensor (result), it shouldn't pick it's version).
#define IMPLEMENT_STATELESS_REVERSED(name)                                     \
static PyObject * TH_CONCAT_2(THPModule_, name)(PyObject *_unused, PyObject *args, PyObject *kwargs) \
{                                                                              \
  PyObject *tensor = THPDefaultTensorClass;                                    \
  PyObject *key, *value;                                                       \
  Py_ssize_t pos = 0;                                                          \
  for (int i = PyTuple_Size(args)-1; i >= 0; i--) {                            \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item)) {                                            \
      tensor = item;                                                           \
      goto dispatch;                                                           \
    }                                                                          \
  }                                                                            \
  if (kwargs) {                                                                \
    while (PyDict_Next(kwargs, &pos, &key, &value)) {                          \
      if (THPModule_isTensor(value)) {                                         \
        tensor = value;                                                        \
        goto dispatch;                                                         \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
dispatch:                                                                      \
  THPObjectPtr methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME); \
  THPUtils_assert(methods, "Type %s doesn't implement stateless methods",      \
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor)); \
  THPObjectPtr method = PyObject_GetAttrString(methods, #name);                \
  THPUtils_assert(method, "Type %s doesn't implement stateless method " #name, \
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor)); \
  return PyObject_Call(method, args, kwargs);                                  \
}

IMPLEMENT_STATELESS_REVERSED(gt)
IMPLEMENT_STATELESS_REVERSED(lt)
IMPLEMENT_STATELESS_REVERSED(ge)
IMPLEMENT_STATELESS_REVERSED(le)
IMPLEMENT_STATELESS_REVERSED(eq)
IMPLEMENT_STATELESS_REVERSED(ne)

#undef IMPLEMENT_STATELESS

// In nonzero, the first argument might be a LongTensor that will be used
// for indices output, so we should pick a function based on second
// tensor's type.
static PyObject * THPModule_nonzero(PyObject *_unused, PyObject *args)
{
  PyObject *tensor = THPDefaultTensorClass;
  if (PyTuple_Size(args) == 1)
    tensor = PyTuple_GET_ITEM(args, 0);
  else if (PyTuple_Size(args) == 2)
    tensor = PyTuple_GET_ITEM(args, 1);

  THPObjectPtr methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME);
  THPUtils_assert(methods, "Type %s doesn't implement stateless methods",
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor));
  THPObjectPtr method = PyObject_GetAttrString(methods, "nonzero");
  THPUtils_assert(method, "Type %s doesn't implement stateless method nonzero",
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor));
  return PyObject_Call(method, args, NULL);
}

static PyObject * THPModule_cat(PyObject *_unused, PyObject *args)
{
  PyObject *tensor = THPDefaultTensorClass;
  THPObjectPtr iterator;
  THPObjectPtr item;
  if (args && PyTuple_Size(args) > 0) {
    if (THPModule_isTensor(PyTuple_GET_ITEM(args, 0))) {
      tensor = PyTuple_GET_ITEM(args, 0);
    } else if ((iterator = PyObject_GetIter(PyTuple_GET_ITEM(args, 0)))) {
      item = PyIter_Next(iterator);
      if (item && THPModule_isTensor(item)) {
        tensor = item;
      }
    }
    PyErr_Clear();
  }

  THPObjectPtr methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME);
  THPUtils_assert(methods, "Type %s doesn't implement statless methods",
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor));
  THPObjectPtr method = PyObject_GetAttrString(methods, "cat");
  THPUtils_assert(method, "Type %s doesn't implement stateless method cat",
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor));
  return PyObject_Call(method, args, NULL);
}

PyObject *THPModule_safeCall(PyObject *_unused, PyObject *args, PyObject *kwargs)
{
  PyObject *result = NULL;
  PyObject *args_slice = NULL;
  PyThreadState *thread_state = PyThreadState_Get();
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;
  THPUtils_assert(num_args > 0, "expected at least one argument");
  try {
    args_slice = PyTuple_GetSlice(args, 1, num_args);
    result = PyObject_Call(PyTuple_GET_ITEM(args, 0), args_slice, kwargs);
  } catch (std::exception &e) {
    PyEval_RestoreThread(thread_state);
    Py_DECREF(args_slice);
    PyErr_SetString(THPException_FatalError, e.what());
    Py_LeaveRecursiveCall();
  }
  Py_DECREF(args_slice);
  return result;
}

#ifdef WITH_CUDA
extern PyObject * THCPModule_initExtension(PyObject *self);
extern PyObject * THCPModule_setDevice_wrap(PyObject *self, PyObject *arg);
extern PyObject * THCPModule_getDevice_wrap(PyObject *self);
extern PyObject * THCPModule_getDeviceCount_wrap(PyObject *self);
extern PyObject * THCPModule_getCurrentStream_wrap(PyObject *self);
extern PyObject * THCPModule_setStream_wrap(PyObject *self, PyObject *stream);
extern PyObject * THCPModule_getDriverVersion(PyObject *self);
extern PyObject * THCPModule_isDriverSufficient(PyObject *self);
extern PyObject * THCPModule_getRNGState(PyObject *_unused);
extern PyObject * THCPModule_setRNGState(PyObject *_unused, PyObject *_new_rng_state);
extern PyObject * THCPModule_manualSeed(PyObject *_unused, PyObject *seed);
extern PyObject * THCPModule_manualSeedAll(PyObject *_unused, PyObject *seed);
extern PyObject * THCPModule_seed(PyObject *_unused);
extern PyObject * THCPModule_seedAll(PyObject *_unused);
extern PyObject * THCPModule_initialSeed(PyObject *_unused);
extern PyObject * THCPModule_cudaHostAllocator(PyObject *_unused);
extern PyObject * THCPModule_cudaSynchronize(PyObject *_unused);
#endif

static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,     METH_O,  NULL},
  {"_autograd_init",  (PyCFunction)THPAutograd_initExtension,   METH_NOARGS,  NULL},
#ifdef WITH_CUDA
  {"_cuda_init",      (PyCFunction)THCPModule_initExtension,    METH_NOARGS,  NULL},
  {"_cuda_setDevice", (PyCFunction)THCPModule_setDevice_wrap,   METH_O,       NULL},
  {"_cuda_getDevice", (PyCFunction)THCPModule_getDevice_wrap,   METH_NOARGS,  NULL},
  {"_cuda_getDeviceCount", (PyCFunction)THCPModule_getDeviceCount_wrap, METH_NOARGS, NULL},
  {"_cuda_getCurrentStream", (PyCFunction)THCPModule_getCurrentStream_wrap, METH_NOARGS, NULL},
  {"_cuda_setStream", (PyCFunction)THCPModule_setStream_wrap, METH_O, NULL},
  {"_cuda_isDriverSufficient", (PyCFunction)THCPModule_isDriverSufficient, METH_NOARGS, NULL},
  {"_cuda_getDriverVersion", (PyCFunction)THCPModule_getDriverVersion, METH_NOARGS, NULL},
  {"_cuda_getRNGState", (PyCFunction)THCPModule_getRNGState, METH_NOARGS, NULL},
  {"_cuda_setRNGState", (PyCFunction)THCPModule_setRNGState, METH_O, NULL},
  {"_cuda_manualSeed", (PyCFunction)THCPModule_manualSeed, METH_O, NULL},
  {"_cuda_manualSeedAll", (PyCFunction)THCPModule_manualSeedAll, METH_O, NULL},
  {"_cuda_seed", (PyCFunction)THCPModule_seed, METH_NOARGS, NULL},
  {"_cuda_seedAll", (PyCFunction)THCPModule_seedAll, METH_NOARGS, NULL},
  {"_cuda_initialSeed", (PyCFunction)THCPModule_initialSeed, METH_NOARGS, NULL},
  {"_cuda_cudaHostAllocator", (PyCFunction)THCPModule_cudaHostAllocator, METH_NOARGS, NULL},
  {"_cuda_synchronize", (PyCFunction)THCPModule_cudaSynchronize, METH_NOARGS, NULL},
#endif
  {"_safe_call",      (PyCFunction)THPModule_safeCall,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"_sendfd",         (PyCFunction)THPModule_sendfd,            METH_VARARGS, NULL},
  {"_recvfd",         (PyCFunction)THPModule_recvfd,            METH_VARARGS, NULL},
  {"_set_default_tensor_type", (PyCFunction)THPModule_setDefaultTensorType, METH_O, NULL},
  {"_tensorCopy",     (PyCFunction)THPModule_tensorCopyWrapper, METH_VARARGS, NULL},
  {"_storageCopy",    (PyCFunction)THPModule_storageCopyWrapper, METH_VARARGS, NULL},
  {"_tensorCopyAsync", (PyCFunction)THPModule_tensor_asyncCopyWrapper, METH_VARARGS, NULL},
  {"_storageCopyAsync", (PyCFunction)THPModule_storage_asyncCopyWrapper, METH_VARARGS, NULL},
  {"get_num_threads", (PyCFunction)THPModule_getNumThreads,     METH_NOARGS,  NULL},
  {"set_num_threads", (PyCFunction)THPModule_setNumThreads,     METH_O,       NULL},
  {"from_numpy",      (PyCFunction)THPModule_fromNumpy,         METH_O,       NULL},

  {"sigmoid",         (PyCFunction)THPModule_sigmoid,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"log",             (PyCFunction)THPModule_log,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"log1p",           (PyCFunction)THPModule_log1p,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"exp",             (PyCFunction)THPModule_exp,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"cos",             (PyCFunction)THPModule_cos,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"acos",            (PyCFunction)THPModule_acos,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cosh",            (PyCFunction)THPModule_cosh,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"sin",             (PyCFunction)THPModule_sin,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"asin",            (PyCFunction)THPModule_asin,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"sinh",            (PyCFunction)THPModule_sinh,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"tan",             (PyCFunction)THPModule_tan,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan",            (PyCFunction)THPModule_atan,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"tanh",            (PyCFunction)THPModule_tanh,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"sqrt",            (PyCFunction)THPModule_sqrt,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"rsqrt",           (PyCFunction)THPModule_rsqrt,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"ceil",            (PyCFunction)THPModule_ceil,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"floor",           (PyCFunction)THPModule_floor,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"round",           (PyCFunction)THPModule_round,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"abs",             (PyCFunction)THPModule_abs,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"trunc",           (PyCFunction)THPModule_trunc,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"frac",            (PyCFunction)THPModule_frac,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"mean",            (PyCFunction)THPModule_mean,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"std",             (PyCFunction)THPModule_std,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"var",             (PyCFunction)THPModule_var,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"norm",            (PyCFunction)THPModule_norm,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cinv",            (PyCFunction)THPModule_cinv,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"neg",             (PyCFunction)THPModule_neg,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"add",             (PyCFunction)THPModule_add,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"csub",            (PyCFunction)THPModule_csub,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul",             (PyCFunction)THPModule_mul,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"div",             (PyCFunction)THPModule_div,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"fmod",            (PyCFunction)THPModule_fmod,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"mod",             (PyCFunction)THPModule_fmod,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cmul",            (PyCFunction)THPModule_cmul,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cdiv",            (PyCFunction)THPModule_cdiv,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cfmod",           (PyCFunction)THPModule_cfmod,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"cmod",            (PyCFunction)THPModule_cfmod,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"min",             (PyCFunction)THPModule_min,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"max",             (PyCFunction)THPModule_max,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"cmax",            (PyCFunction)THPModule_cmax,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cmin",            (PyCFunction)THPModule_cmin,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpow",            (PyCFunction)THPModule_cpow,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"dot",             (PyCFunction)THPModule_dot,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"sum",             (PyCFunction)THPModule_sum,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"prod",            (PyCFunction)THPModule_prod,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"remainder",       (PyCFunction)THPModule_remainder,         METH_VARARGS | METH_KEYWORDS, NULL},
  {"cremainder",      (PyCFunction)THPModule_cremainder,        METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumsum",          (PyCFunction)THPModule_cumsum,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumprod",         (PyCFunction)THPModule_cumprod,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp",           (PyCFunction)THPModule_clamp,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"equal",           (PyCFunction)THPModule_equal,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"eye",             (PyCFunction)THPModule_eye,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"fill",            (PyCFunction)THPModule_fill,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"diag",            (PyCFunction)THPModule_diag,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"numel",           (PyCFunction)THPModule_numel,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"sign",            (PyCFunction)THPModule_sign,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"trace",           (PyCFunction)THPModule_trace,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril",            (PyCFunction)THPModule_tril,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu",            (PyCFunction)THPModule_triu,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"zero",            (PyCFunction)THPModule_zero,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"gt",              (PyCFunction)THPModule_gt,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt",              (PyCFunction)THPModule_lt,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge",              (PyCFunction)THPModule_ge,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"le",              (PyCFunction)THPModule_le,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq",              (PyCFunction)THPModule_eq,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne",              (PyCFunction)THPModule_ne,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"kthvalue",        (PyCFunction)THPModule_kthvalue,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"mode",            (PyCFunction)THPModule_mode,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"median",          (PyCFunction)THPModule_median,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"cross",           (PyCFunction)THPModule_cross,             METH_VARARGS | METH_KEYWORDS, NULL},
  //{"sort",            (PyCFunction)THPModule_sort,              METH_VARARGS | METH_KEYWORDS, NULL},
  //{"topk",            (PyCFunction)THPModule_topk,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"t",               (PyCFunction)THPModule_t,                 METH_VARARGS | METH_KEYWORDS, NULL},
  {"transpose",       (PyCFunction)THPModule_transpose,         METH_VARARGS | METH_KEYWORDS, NULL},
  {"squeeze",         (PyCFunction)THPModule_squeeze,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"nonzero",         (PyCFunction)THPModule_nonzero,           METH_VARARGS, NULL},
  {"renorm",          (PyCFunction)THPModule_renorm,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"dist",            (PyCFunction)THPModule_dist,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"linspace",        (PyCFunction)THPModule_linspace,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"logspace",        (PyCFunction)THPModule_logspace,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"histc",           (PyCFunction)THPModule_histc,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan2",           (PyCFunction)THPModule_atan2,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow",             (PyCFunction)THPModule_pow,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp",            (PyCFunction)THPModule_lerp,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"reshape",         (PyCFunction)THPModule_reshape,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"zeros",           (PyCFunction)THPModule_zeros,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"ones",            (PyCFunction)THPModule_ones,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_select",    (PyCFunction)THPModule_index_select,      METH_VARARGS | METH_KEYWORDS, NULL},
  {"narrow",          (PyCFunction)THPModule_narrow,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm",           (PyCFunction)THPModule_addmm,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv",           (PyCFunction)THPModule_addmv,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr",            (PyCFunction)THPModule_addr,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"ger",             (PyCFunction)THPModule_ger,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"mv",              (PyCFunction)THPModule_mv,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm",          (PyCFunction)THPModule_addbmm,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"baddbmm",         (PyCFunction)THPModule_baddbmm,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul",         (PyCFunction)THPModule_addcmul,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv",         (PyCFunction)THPModule_addcdiv,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"mm",              (PyCFunction)THPModule_mm,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"bmm",             (PyCFunction)THPModule_bmm,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"multinomial",     (PyCFunction)THPModule_multinomial,       METH_VARARGS | METH_KEYWORDS, NULL},
  {"uniform",         (PyCFunction)THPModule_uniform,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"normal",          (PyCFunction)THPModule_normal,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"cauchy",          (PyCFunction)THPModule_cauchy,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"log_normal",      (PyCFunction)THPModule_log_normal,        METH_VARARGS | METH_KEYWORDS, NULL},
  {"exponential",     (PyCFunction)THPModule_exponential,       METH_VARARGS | METH_KEYWORDS, NULL},
  {"random",          (PyCFunction)THPModule_random,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"geometric",       (PyCFunction)THPModule_geometric,         METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli",       (PyCFunction)THPModule_bernoulli,         METH_VARARGS | METH_KEYWORDS, NULL},
  {"rand",            (PyCFunction)THPModule_rand,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"randn",           (PyCFunction)THPModule_randn,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"randperm",        (PyCFunction)THPModule_randperm,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"unfold",          (PyCFunction)THPModule_unfold,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"range",           (PyCFunction)THPModule_range,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"gather",          (PyCFunction)THPModule_gather,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter",         (PyCFunction)THPModule_scatter,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"all",             (PyCFunction)THPModule_all,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"any",             (PyCFunction)THPModule_any,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"cat",             (PyCFunction)THPModule_cat,               METH_VARARGS, NULL},
  {"masked_select",   (PyCFunction)THPModule_masked_select,     METH_VARARGS | METH_KEYWORDS, NULL},
  {"gesv",            (PyCFunction)THPModule_gesv,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"gels",            (PyCFunction)THPModule_gels,              METH_VARARGS | METH_KEYWORDS, NULL},
  {"trtrs",           (PyCFunction)THPModule_trtrs,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"symeig",          (PyCFunction)THPModule_symeig,            METH_VARARGS | METH_KEYWORDS, NULL},
  {"eig",             (PyCFunction)THPModule_eig,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"svd",             (PyCFunction)THPModule_svd,               METH_VARARGS | METH_KEYWORDS, NULL},
  {"inverse",         (PyCFunction)THPModule_inverse,           METH_VARARGS | METH_KEYWORDS, NULL},
  {"potrf",           (PyCFunction)THPModule_potrf,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"potrs",           (PyCFunction)THPModule_potrs,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"potri",           (PyCFunction)THPModule_potri,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"pstrf",           (PyCFunction)THPModule_pstrf,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"qr",              (PyCFunction)THPModule_qr,                METH_VARARGS | METH_KEYWORDS, NULL},
  {"geqrf",           (PyCFunction)THPModule_geqrf,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"orgqr",           (PyCFunction)THPModule_orgqr,             METH_VARARGS | METH_KEYWORDS, NULL},
  {"ormqr",           (PyCFunction)THPModule_ormqr,             METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL, NULL, 0, NULL}
};

static void errorHandler(const char *msg, void *data)
{
  throw THException(msg);
}

static void errorHandlerArg(int argNumber, const char *msg, void *data)
{
  throw THArgException(msg, argNumber);
}

static void updateErrorHandlers()
{
  THSetDefaultErrorHandler(errorHandler, NULL);
  THSetDefaultArgErrorHandler(errorHandlerArg, NULL);
}

bool THCPDoubleStorage_init(PyObject *module);
bool THCPFloatStorage_init(PyObject *module);
bool THCPHalfStorage_init(PyObject *module);
bool THCPLongStorage_init(PyObject *module);
bool THCPIntStorage_init(PyObject *module);
bool THCPShortStorage_init(PyObject *module);
bool THCPCharStorage_init(PyObject *module);
bool THCPByteStorage_init(PyObject *module);

bool THCPDoubleTensor_init(PyObject *module);
bool THCPFloatTensor_init(PyObject *module);
bool THCPHalfTensor_init(PyObject *module);
bool THCPLongTensor_init(PyObject *module);
bool THCPIntTensor_init(PyObject *module);
bool THCPShortTensor_init(PyObject *module);
bool THCPCharTensor_init(PyObject *module);
bool THCPByteTensor_init(PyObject *module);

bool THCPStream_init(PyObject *module);

static std::vector<PyMethodDef> methods;

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C()
#else
PyMODINIT_FUNC PyInit__C()
#endif
{

#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif

  THPUtils_addPyMethodDefs(methods, TorchMethods);
#ifdef WITH_CUDNN
  THPUtils_addPyMethodDefs(methods, THCUDNN_methods());
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", methods.data()));
#else
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     NULL,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));
  ASSERT_TRUE(THPSize_init(module));
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));

  ASSERT_TRUE(THPDoubleStorage_init(module));
  ASSERT_TRUE(THPFloatStorage_init(module));
  ASSERT_TRUE(THPLongStorage_init(module));
  ASSERT_TRUE(THPIntStorage_init(module));
  ASSERT_TRUE(THPShortStorage_init(module));
  ASSERT_TRUE(THPCharStorage_init(module));
  ASSERT_TRUE(THPByteStorage_init(module));

  ASSERT_TRUE(THPDoubleTensor_init(module));
  ASSERT_TRUE(THPFloatTensor_init(module));
  ASSERT_TRUE(THPLongTensor_init(module));
  ASSERT_TRUE(THPIntTensor_init(module));
  ASSERT_TRUE(THPShortTensor_init(module));
  ASSERT_TRUE(THPCharTensor_init(module));
  ASSERT_TRUE(THPByteTensor_init(module));

#ifdef WITH_CUDA
  // This will only initialise base classes and attach them to library namespace
  // They won't be ready for real usage until importing cuda module, that will
  // complete the process (but it defines Python classes before calling back into
  // C, so these lines have to execute first)..
  ASSERT_TRUE(THCPDoubleStorage_init(module));
  ASSERT_TRUE(THCPFloatStorage_init(module));
  ASSERT_TRUE(THCPHalfStorage_init(module));
  ASSERT_TRUE(THCPLongStorage_init(module));
  ASSERT_TRUE(THCPIntStorage_init(module));
  ASSERT_TRUE(THCPShortStorage_init(module));
  ASSERT_TRUE(THCPCharStorage_init(module));
  ASSERT_TRUE(THCPByteStorage_init(module));
  ASSERT_TRUE(THCPHalfStorage_init(module));

  ASSERT_TRUE(THCPDoubleTensor_init(module));
  ASSERT_TRUE(THCPFloatTensor_init(module));
  ASSERT_TRUE(THCPHalfTensor_init(module));
  ASSERT_TRUE(THCPLongTensor_init(module));
  ASSERT_TRUE(THCPIntTensor_init(module));
  ASSERT_TRUE(THCPShortTensor_init(module));
  ASSERT_TRUE(THCPCharTensor_init(module));
  ASSERT_TRUE(THCPByteTensor_init(module));

  ASSERT_TRUE(THCPStream_init(module));
#endif

#ifdef WITH_CUDNN
  ASSERT_TRUE(THCUDNNModule_initModule(module));
#endif

  THPDefaultGenerator = (THPGenerator*)THPGenerator_New();
  ASSERT_TRUE(PyModule_AddObject(module, "default_generator", (PyObject*)THPDefaultGenerator) == 0);
  ASSERT_TRUE(THPDefaultGenerator != nullptr);

  updateErrorHandlers();

#ifdef WITH_NUMPY
  import_array();
#endif

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif

#undef ASSERT_TRUE
}
