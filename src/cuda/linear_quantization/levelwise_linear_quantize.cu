#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/linear_quantization.h"
#include "cuda/linear_quantization.hpp"

namespace mgard_cuda {

#define KERNELS(T, D) \
        template void levelwise_linear_quantize<T, D>(\
		mgard_cuda_handle<T, D> &handle, \
        int * shapes, int l_target,\
         quant_meta<T> m, \
         T *dv, int * ldvs,\ 
         int *dwork, int * ldws,\ 
         bool prep_huffmam,\
         int * shape,\
         size_t * outlier_count, unsigned int * outlier_idx, int * outliers,\
        int queue_idx);

KERNELS(double, 1)
KERNELS(float,  1)
KERNELS(double, 2)
KERNELS(float,  2)
KERNELS(double, 3)
KERNELS(float,  3)
KERNELS(double, 4)
KERNELS(float,  4)
KERNELS(double, 5)
KERNELS(float,  5)

#undef KERNELS



}