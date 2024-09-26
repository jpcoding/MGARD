/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../Utilities/Types.h"

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include "Compressor.h"
#include "CompressorCache.hpp"

#include "mgard-x/RuntimeX/DataTypes.h"
#include "reorder_int.hpp"
#include "quant_pred.hpp"


#ifndef MGARD_X_COMPRESSOR_HPP
#define MGARD_X_COMPRESSOR_HPP

namespace mgard_x {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
Compressor<D, T, DeviceType>::Compressor() : initialized(false) {}

template <DIM D, typename T, typename DeviceType>
Compressor<D, T, DeviceType>::Compressor(Hierarchy<D, T, DeviceType> &hierarchy,
                                         Config config)
    : initialized(true), hierarchy(&hierarchy), config(config),
      refactor(hierarchy, config),
      lossless_compressor(hierarchy.total_num_elems(), config),
      quantizer(hierarchy, config) {

  norm_array = Array<1, T, DeviceType>({1});
  // Reuse workspace. Warning:
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    // Reuse workspace if possible
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                             (T *)refactor.w_array.data());
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()),
        (QUANTIZED_INT *)refactor.w_array.data());
  } else {
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()});
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()), false, false);
  }
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Adapt(Hierarchy<D, T, DeviceType> &hierarchy,
                                         Config config, int queue_idx) {
  this->initialized = true;
  this->hierarchy = &hierarchy;
  this->config = config;
  refactor.Adapt(hierarchy, config, queue_idx);
  lossless_compressor.Adapt(hierarchy.total_num_elems(), config, queue_idx);
  quantizer.Adapt(hierarchy, config, queue_idx);
  norm_array.resize({1}, queue_idx);
  // Reuse workspace. Warning:
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    // Reuse workspace if possible
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                             (T *)refactor.w_array.data());
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()),
        (QUANTIZED_INT *)refactor.w_array.data());
  } else {
    norm_tmp_array.resize({hierarchy.total_num_elems()}, queue_idx);
    quantized_array.resize(hierarchy.level_shape(hierarchy.l_target()),
                           queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
size_t
Compressor<D, T, DeviceType>::EstimateMemoryFootprint(std::vector<SIZE> shape,
                                                      Config config) {
  Hierarchy<D, T, DeviceType> hierarchy;
  hierarchy.EstimateMemoryFootprint(shape);
  size_t size = 0;
  size += DataRefactorType::EstimateMemoryFootprint(shape);
  size += LinearQuantizerType::EstimateMemoryFootprint(shape);
  size += LosslessCompressorType::EstimateMemoryFootprint(
      hierarchy.total_num_elems(), config);
  size += sizeof(T);
  if (sizeof(QUANTIZED_INT) > sizeof(T)) {
    size += sizeof(T) * hierarchy.total_num_elems();
    size += sizeof(QUANTIZED_INT) * hierarchy.total_num_elems();
  }
  return size;
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::CalculateNorm(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T s,
    T &norm, int queue_idx) {
  if (ebtype == error_bound_type::REL) {
    norm =
        norm_calculator(original_data, SubArray(norm_tmp_array),
                        SubArray(norm_array), s, config.normalize_coordinates);
  }
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Decompose(
    Array<D, T, DeviceType> &original_data, int queue_idx) {
  refactor.Decompose(SubArray(original_data), queue_idx);
}

  template<typename Type>
  void writefile_(const char *file, Type *data, size_t num_elements) {
      std::ofstream fout(file, std::ios::binary);
      fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
      fout.close();
  }


template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Quantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {

  quantizer.Quantize(original_data, ebtype, tol, s, norm, quantized_array,
                     lossless_compressor, queue_idx);


  // std::cout << "dict size: " << quantizer.get_dict_size() << std::endl; 

  // reorder the quntization index;
  // for(auto i = 0; i < quantized_array.shape().size(); i++){
  //   std::cout << quantized_array.shape()[i] << " ";
  // }
  // writefile_("quant_inds.dat", quantized_array.data(), quantized_array.totalNumElems());

  // reverse the order of the quantized array
  if constexpr (D == 3){
  // writefile_("quant_inds_q_1.dat", quantized_array.data(), quantized_array.totalNumElems());

  if(1){

      std::vector<size_t> dims;
      for(auto i = 0; i < quantized_array.shape().size(); i++){
        dims.push_back(quantized_array.shape()[i]);
      }

      int max_level = log2(*min_element(dims.begin(), dims.end())) - 1;
      int target_level = 4; 
      std::vector<std::vector<size_t>> level_dims; 
      for (int i = 0; i <= target_level; i++) {
      level_dims.push_back(std::vector<size_t>(dims.size()));
      }
      for (int i = 0; i < dims.size(); i++) {
        int n = dims[i];
        for (int j = 0; j <= target_level; j++) {
          level_dims[target_level - j][i] = n;
          n = (n >> 1) + 1;
        }
      }
      // for(int i = 0; i< level_dims.size(); i++){
      //     std::cout << "level " << i << " : " << level_dims[i][0] << " " << level_dims[i][1] << " " << level_dims[i][2] << std::endl;
      // }
      // std::cout << "target_level = " << target_level << std::endl;    

      std::vector<QUANTIZED_INT> data_buffer(quantized_array.totalNumElems());
      for(int i=0; i<target_level; i++){
          size_t n1 = level_dims[i+1][0];
          size_t n2 = level_dims[i+1][1];
          size_t n3 = level_dims[i+1][2];
          // printf("level = %d, n1 = %ld , n2 = %ld, n3 = %ld\n", i,n1, n2, n3);
          MGARD_INT::data_reverse_reorder_3D(quantized_array.data(), 
                      data_buffer.data(), n1, n2, n3, 
                      dims[1]*dims[2], dims[2]);
      }
      // writefile_("quant_inds_q_2.dat", quantized_array.data(), quantized_array.totalNumElems());

    }

 
    if(1) {
      std::vector<size_t> dims;
      for(auto i = 0; i < quantized_array.shape().size(); i++){
        dims.push_back(quantized_array.shape()[i]);
        // std::cout << dims[i] << " ";
      }
      // int radius = quantizer.get_dict_size()/2;
      int radius = config.huff_dict_size/2;
      // std::cout << "dict size: " << quantizer.get_dict_size() << std::endl;
      // std::cout << "radius = " << radius << std::endl;


      auto quantization_predicter = MGARD::QuantPred<QUANTIZED_INT>(quantized_array.data(), 
                      3, dims.data(), 2, 0, radius);
      quantization_predicter.quant_pred_level_3D(1);
      quantization_predicter.quant_pred_level_3D(2);

      auto quant_min = *std::min_element(quantized_array.data(), quantized_array.data() + quantized_array.totalNumElems());
      auto quant_max = *std::max_element(quantized_array.data(), quantized_array.data() + quantized_array.totalNumElems());

      // 
      // std::cout << "quant_min = " << quant_min << std::endl;
      // std::cout << "quant_max = " << quant_max << std::endl;
      // writefile_("quant_inds_q_3.dat", quantized_array.data(), quantized_array.totalNumElems());


    }
    // inplace check the quantization prediction
    if(0)
    {
      std::vector<size_t> dims;
      for(auto i = 0; i < quantized_array.shape().size(); i++){
        dims.push_back(quantized_array.shape()[i]);
        std::cout << dims[i] << " ";
      }
      std::vector<QUANTIZED_INT> quant_inds_copy(quantized_array.totalNumElems());
      std::copy(quantized_array.data(), quantized_array.data() + quantized_array.totalNumElems(), quant_inds_copy.data());
      int radius = quantizer.get_dict_size()/2; 
      std::cout << "dict size: " << quantizer.get_dict_size() << std::endl;
      auto quantization_predicter = MGARD::QuantPred<QUANTIZED_INT>(quant_inds_copy.data(), 
                      3, dims.data(), 2, 0, radius, 0);			
      quantization_predicter.quant_pred_level_3D_recover(1);
			quantization_predicter.quant_pred_level_3D_recover(2);
      // writefile_("quant_inds_pred_recover_check.dat", quant_inds_copy.data(), quant_inds_copy.size());
    }
  }


}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::LosslessCompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  // std::cout << "point1\n";
  // Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> quantized_liearized_array(
  //     {hierarchy->total_num_elems()},
  //     (QUANTIZED_UNSIGNED_INT *)quantized_array.data()); // A simple type conversion ? problem
  Array<1, QUANTIZED_INT, DeviceType> quantized_liearized_array(
    {hierarchy->total_num_elems()},
    (QUANTIZED_INT *)quantized_array.data()); // A simple type conversion ? problem

  lossless_compressor.Compress(quantized_liearized_array, compressed_data,
                               queue_idx);
  // std::cout << "point3\n";
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Recompose(
    Array<D, T, DeviceType> &decompressed_data, int queue_idx) {
  refactor.Recompose(SubArray(decompressed_data), queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Dequantize(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));



  if constexpr (D ==3)
  {
    std::array<size_t,3> dims;
    for(auto i = 0; i < quantized_array.shape().size(); i++){
        dims[i]= (quantized_array.shape()[i]);
    }
    // recover the quantization prediction

    if(1)
    {
      int radius = config.huff_dict_size/2;
      // writefile_("quant_inds_d_3.dat", quantized_array.data(), quantized_array.totalNumElems());

      // std::cout << "dict size: " << quantizer.get_dict_size() << std::endl;

      auto quantization_predicter = MGARD::QuantPred<QUANTIZED_INT>(quantized_array.data(), 
                      3, dims.data(), 2, 0, radius,0);			
      quantization_predicter.quant_pred_level_3D_recover(1);
			quantization_predicter.quant_pred_level_3D_recover(2);

    }


    // reorder the quntization index; 

    if(1)
    {
      // writefile_("quant_inds_d_2.dat", quantized_array.data(), quantized_array.totalNumElems());
      size_t n1 = dims[0];
      size_t n2 = dims[1];
      size_t n3 = dims[2];
      int max_level = log2(*std::min_element(dims.begin(), dims.end())) - 1;
      int target_level =4; 
      std::vector<QUANTIZED_INT> data_buffer(quantized_array.totalNumElems());
      // std::cout << "target_level = " << target_level << std::endl;    
      for(int i=0; i<target_level; i++){
          // printf("level = %d, n1 = %ld , n2 = %ld, n3 = %ld\n", i, n1, n2, n3);
          MGARD_INT::data_reorder_3D(quantized_array.data(), 
          data_buffer.data(), n1, n2, n3, dims[1]*dims[2], dims[2]);
          n1 = (n1 >> 1) + 1;
          n2 = (n2 >> 1) + 1;
          n3 = (n3 >> 1) + 1;
      }
    }
    // writefile_("quant_inds_d_1.dat", quantized_array.data(), quantized_array.totalNumElems());


  }
  // std::cout << "huff dict size " << config.huff_dict_size << std::endl;
  quantizer.Dequantize(decompressed_data, ebtype, tol, s, norm, quantized_array,
                       lossless_compressor, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::LosslessDecompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  Array<1, QUANTIZED_INT, DeviceType> quantized_liearized_data(
      {hierarchy->total_num_elems()},
      (QUANTIZED_INT *)quantized_array.data());
  // Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> quantized_liearized_data(
  //     {hierarchy->total_num_elems()},
  //     (QUANTIZED_UNSIGNED_INT *)quantized_array.data());
  lossless_compressor.Decompress(compressed_data, quantized_liearized_data,
                                 queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Compress(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T &norm, Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy->level_shape(hierarchy->l_target(), d) !=
        original_data.shape(d)) {
      log::err("The shape of input array does not match the shape initialized "
               "in hierarchy!");
      return;
    }
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.start();
  }

  CalculateNorm(original_data, ebtype, s, norm, queue_idx);
  Decompose(original_data, queue_idx);
  Quantize(original_data, ebtype, tol, s, norm, queue_idx);
  LosslessCompress(compressed_data, queue_idx);
  if (config.compress_with_dryrun) {
    Dequantize(original_data, ebtype, tol, s, norm, queue_idx);
    Recompose(original_data, queue_idx);
    // writefile_("decompressed_mgard_pred.out", original_data.data(), original_data.totalNumElems());
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.end();
    timer_total.print("Low-level compression");
    log::time(
        "Low-level compression throughput: " +
        std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                       timer_total.get() / 1e9) +
        " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Decompress(
    Array<1, Byte, DeviceType> &compressed_data, enum error_bound_type ebtype,
    T tol, T s, T &norm, Array<D, T, DeviceType> &decompressed_data,
    int queue_idx) {
  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.start();
  }

  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  LosslessDecompress(compressed_data, queue_idx);
  std::cout << "dict size: " << quantizer.get_dict_size() << std::endl;
  std::cout << "huffman dict size " << config.huff_dict_size << std::endl;
  Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  Recompose(decompressed_data, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.end();
    timer_total.print("Low-level decompression");
    log::time(
        "Low-level decompression throughput: " +
        std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                       timer_total.get() / 1e9) +
        " GB/s");
    timer_total.clear();
  }
}

} // namespace mgard_x

#endif