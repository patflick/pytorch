#ifndef THC_ATOMICS_INC
#define THC_ATOMICS_INC

#include "THCHalf.h"
#include "THCNumerics.cuh"

#if defined (__HIP_PLATFORM_NVCC__)
template <typename T, size_t n>
struct AtomicAddIntegerImpl;

template<typename T>
struct AtomicAddIntegerImpl<T, 1> {
  inline __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui =
        (unsigned int *) (address - ((size_t)address & 3));
    unsigned int old = *address_as_ui;
    unsigned int shift = (((size_t)address & 3) * 8);
    unsigned int sum;
    unsigned int assumed;

    do {
      assumed = old;
      sum = val + T((old >> shift) & 0xff);
      old = (old & ~(0x000000ff << shift)) | (sum << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui =
        (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int sum;
    unsigned int newval;
    unsigned int assumed;

    do {
      assumed = old;
      sum = val + (size_t)address & 2 ? T(old >> 16) : T(old & 0xffff);
      newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16) : (old & 0xffff0000) | sum;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 4> {
  inline __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui = (unsigned int *) (address);
    unsigned int old = *address_as_ui;
    unsigned int newval;
    unsigned int assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 8> {
  inline __device__ void operator()(T *address, T val) {
    unsigned long long * address_as_ui = (unsigned long long *) (address);
    unsigned long long old = *address_as_ui;
    unsigned long long newval;
    unsigned long long assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

static inline __device__ void atomicAdd(unsigned char *address, unsigned char val) {
  AtomicAddIntegerImpl<unsigned char, sizeof(unsigned char)>()(address, val);
}

static inline  __device__ void atomicAdd(char *address, char val) {
  AtomicAddIntegerImpl<char, sizeof(char)>()(address, val);
}

static inline  __device__ void atomicAdd(short *address, short val) {
  AtomicAddIntegerImpl<short, sizeof(short)>()(address, val);
}

static inline __device__ void atomicAdd(long *address, long val) {
  AtomicAddIntegerImpl<long, sizeof(long)>()(address, val);
}

#ifdef CUDA_HALF_TENSOR
static inline  __device__ void atomicAdd(half *address, half val) {
  unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = THCNumerics<half>::add(hsum, val);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
   } while (assumed != old);
}
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
// from CUDA C Programmic Guide
static inline  __device__  void atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}
#elif !defined(__CUDA_ARCH__) && (CUDA_VERSION < 8000)
// This needs to be defined for the host side pass
static inline  __device__  void atomicAdd(double *address, double val) { }
#endif



#elif defined (__HIP_PLATFORM_HCC__)
template <typename T, size_t n>
struct AtomicAddIntegerImpl;

template<typename T>
struct AtomicAddIntegerImpl<T, 1> {
  inline __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui =
        (unsigned int *) (address - ((size_t)address & 3));
    unsigned int old = *address_as_ui;
    unsigned int shift = (((size_t)address & 3) * 8);
    unsigned int sum;
    unsigned int assumed;

    do {
      assumed = old;
      sum = val + T((old >> shift) & 0xff);
      old = (old & ~(0x000000ff << shift)) | (sum << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui =
        (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int sum;
    unsigned int newval;
    unsigned int assumed;

    do {
      assumed = old;
      sum = val + (size_t)address & 2 ? T(old >> 16) : T(old & 0xffff);
      newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16) : (old & 0xffff0000) | sum;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 4> {
  inline __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui = (unsigned int *) (address);
    unsigned int old = *address_as_ui;
    unsigned int newval;
    unsigned int assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 8> {
  inline __device__ void operator()(T *address, T val) {
    unsigned long long * address_as_ui = (unsigned long long *) (address);
    unsigned long long old = *address_as_ui;
    unsigned long long newval;
    unsigned long long assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

static inline __device__ void atomicAdd(unsigned char *address, unsigned char val) {
  AtomicAddIntegerImpl<unsigned char, sizeof(unsigned char)>()(address, val);
}

static inline  __device__ void atomicAdd(char *address, char val) {
  AtomicAddIntegerImpl<char, sizeof(char)>()(address, val);
}

static inline  __device__ void atomicAdd(short *address, short val) {
  AtomicAddIntegerImpl<short, sizeof(short)>()(address, val);
}

static inline __device__ void atomicAdd(long *address, long val) {
  AtomicAddIntegerImpl<long, sizeof(long)>()(address, val);
}

#ifdef CUDA_HALF_TENSOR
  #if defined(__HIP_PLATFORM_HCC__)
    __device__
    static
    inline
    void atomicAdd(half* address, half val)
    {
      // TODO: unimplemented.
//        unsigned short* address_as_us =
//            reinterpret_cast<unsigned short*>(address);
//        half old_x = *address;
//        half new_x;
//        do {
//            new_x = old_x + val;
//        } while (!hc::atomic_compare_exchange(
//            address_as_us, reinterpret_cast<unsigned short*>(&old_x), *reinterpret_cast<unsigned short*>(&new_x)));
    }
  #else
    __device__
    static
    inline
    void atomicAdd(half *address, half val)
    {
      unsigned int * address_as_ui =
          (unsigned int *) ((char *)address - ((size_t)address & 2));
      unsigned int old = *address_as_ui;
      unsigned int assumed;

      do {
        assumed = old;
        half hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        hsum = THCNumerics<half>::add(hsum, val);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
       } while (assumed != old);
    }
  #endif
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 || defined(__HIP_PLATFORM_HCC__)
// from CUDA C Programmic Guide
static inline  __device__  void atomicAdd(double *address, double val) {
  uint64_t* address_as_ull = (uint64_t*)address;
//unsigned long long int* address_as_ull = (unsigned long long int*)address;
//  unsigned long long int old = *address_as_ull;
//  unsigned long long int assumed;
//
//  do {
//    assumed = old;
//    //old = atomicCAS(address_as_ull, assumed,
//    //                __double_as_longlong(val +
//    //                __longlong_as_double(assumed)));
//    //old = atomicCAS(address_as_ull, assumed,
//    //                (unsigned long long)(val +
//    //                (double)(assumed)));
//    //double newVal = val + *reinterpret_cast<double*>(&assumed);
//    //old = atomicCAS(address_as_ull, assumed,
//    //                *reinterpret_cast<unsigned long long*>(&newVal));
//
//    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//  } while (assumed != old);

  double old_x = *address;
  double new_x;
  do {
      new_x = old_x + val;
  } while (!hc::atomic_compare_exchange(address_as_ull, reinterpret_cast<uint64_t*>(&old_x), *reinterpret_cast<uint64_t*>(&new_x)));
}
#endif
#endif // HIP_PLATFORM_NVCC
#endif // THC_ATOMICS_INC
