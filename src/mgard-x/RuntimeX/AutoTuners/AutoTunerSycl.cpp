#include "mgard-x/RuntimeX/RuntimeX.h"
// clang-format off
namespace mgard_x {

int AutoTuningTable<SYCL>::gpk_reo_3d[2][9] = {{3, 3, 3, 3, 3, 3, 3, 0, 0},
                                              {3, 5, 5, 4, 3, 3, 4, 0, 0}};

int AutoTuningTable<SYCL>::gpk_rev_3d[2][9] = {{1, 4, 4, 3, 5, 5, 5, 0, 0},
                                              {1, 3, 5, 4, 3, 3, 5, 0, 0}};

int AutoTuningTable<SYCL>::gpk_reo_nd[2][9] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                              {0, 0, 3, 4, 5, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::gpk_rev_nd[2][9] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                              {0, 0, 3, 4, 5, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::lpk1_3d[2][9] = {{4, 4, 3, 0, 1, 2, 4, 0, 0},
                                           {2, 0, 0, 2, 1, 0, 4, 0, 0}};

int AutoTuningTable<SYCL>::lpk2_3d[2][9] = {{0, 1, 0, 0, 3, 5, 4, 0, 0},
                                           {4, 1, 1, 1, 3, 3, 4, 0, 0}};

int AutoTuningTable<SYCL>::lpk3_3d[2][9] = {{2, 2, 0, 3, 1, 4, 5, 0, 0},
                                           {2, 1, 1, 1, 3, 4, 4, 0, 0}};

int AutoTuningTable<SYCL>::lpk1_nd[2][9] = {{2, 0, 1, 1, 1, 0, 0, 0, 0},
                                           {0, 0, 1, 1, 1, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::lpk2_nd[2][9] = {{2, 1, 3, 1, 0, 0, 0, 0, 0},
                                           {0, 2, 1, 1, 0, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::lpk3_nd[2][9] = {{2, 3, 1, 1, 0, 0, 0, 0, 0},
                                           {0, 2, 1, 1, 0, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::ipk1_3d[2][9] = {{1, 4, 1, 4, 3, 4, 5, 0, 0},
                                           {2, 2, 3, 0, 6, 3, 3, 0, 0}};

int AutoTuningTable<SYCL>::ipk2_3d[2][9] = {{1, 2, 1, 2, 2, 2, 6, 0, 0},
                                           {1, 2, 1, 3, 2, 2, 6, 0, 0}};

int AutoTuningTable<SYCL>::ipk3_3d[2][9] = {{1, 2, 2, 2, 2, 2, 6, 0, 0},
                                           {1, 2, 3, 1, 2, 2, 6, 0, 0}};

int AutoTuningTable<SYCL>::ipk1_nd[2][9] = {{0, 2, 3, 3, 0, 0, 0, 0, 0},
                                           {0, 3, 3, 3, 0, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::ipk2_nd[2][9] = {{0, 1, 2, 2, 0, 0, 0, 0, 0},
                                           {0, 2, 2, 2, 0, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::ipk3_nd[2][9] = {{0, 2, 3, 2, 0, 0, 0, 0, 0},
                                           {0, 3, 4, 2, 0, 0, 0, 0, 0}};

int AutoTuningTable<SYCL>::lwpk[2][9] = {{1, 4, 0, 0, 0, 4, 4, 0, 0},
                                        {2, 0, 4, 1, 2, 2, 5, 0, 0}};

int AutoTuningTable<SYCL>::lwqzk[2][9] = {{5, 2, 2, 1, 0, 2, 0, 0, 0},
                                         {2, 2, 3, 3, 0, 2, 0, 0, 0}};

int AutoTuningTable<SYCL>::lwdqzk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                         {2, 2, 3, 3, 0, 2, 5, 0, 0}};  

int AutoTuningTable<SYCL>::llk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                         {2, 2, 3, 3, 0, 2, 5, 0, 0}};

template void BeginAutoTuning<SYCL>();
template void EndAutoTuning<SYCL>();

} // namespace mgard_x
// clang-format on
#undef MGARDX_COMPILE_SYCL