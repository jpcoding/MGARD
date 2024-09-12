/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HUFFMAN_TEMPLATE_HPP
#define MGARD_X_HUFFMAN_TEMPLATE_HPP

#include "mgard-x/Utilities/Types.h"
static bool debug_print_huffman = true;

#include "../LosslessCompressorInterface.hpp"

#include "../../RuntimeX/Utilities/Serializer.hpp"
#include "Condense.hpp"
#include "Decode.hpp"
#include "Deflate.hpp"
#include "DictionaryShift.hpp"
#include "EncodeFixedLen.hpp"
#include "GetCodebook.hpp"
#include "Histogram.hpp"
#include "HuffmanWorkspace.hpp"
#include "OutlierSeparator.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/utils/MemoryUtil.hpp"

#include <chrono>
using namespace std::chrono;

  template<typename Type>
  void writefile__(const char *file, Type *data, size_t num_elements) {
      std::ofstream fout(file, std::ios::binary);
      fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
      fout.close();
  }

namespace mgard_x {

template <typename Q, typename S, typename H, typename DeviceType>
class Huffman : public LosslessCompressorInterface<S, DeviceType> {
public:
  Huffman() : initialized(false) {}

  Huffman(SIZE max_size, int dict_size, int chunk_size,
          double estimated_outlier_ratio)
      : initialized(true), max_size(max_size), dict_size(dict_size),
        chunk_size(chunk_size) {
    workspace = HuffmanWorkspace<Q, S, H, DeviceType>(
        max_size, dict_size, chunk_size, estimated_outlier_ratio);
  }

  void Resize(SIZE max_size, int dict_size, int chunk_size,
              double estimated_outlier_ratio, int queue_idx) {
    this->initialized = true;
    this->max_size = max_size;
    this->dict_size = dict_size;
    this->chunk_size = chunk_size;
    workspace.resize(max_size, dict_size, chunk_size, estimated_outlier_ratio,
                     queue_idx);
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, SIZE dict_size,
                                        SIZE chunk_size,
                                        double estimated_outlier_ratio = 1) {
    return HuffmanWorkspace<Q, S, H, DeviceType>::EstimateMemoryFootprint(
        primary_count, dict_size, chunk_size, estimated_outlier_ratio);
  }

  void CompressPrimary(Array<1, Q, DeviceType> &primary_data,
                       Array<1, Byte, DeviceType> &compressed_data,
                       int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray primary_subarray(primary_data);
    workspace.reset(queue_idx);

    size_t primary_count = primary_subarray.shape(0);

    // Histogram<Q, unsigned int, DeviceType>(primary_subarray,
    //                                        workspace.freq_subarray,
    //                                        primary_count, dict_size, queue_idx);

    // if (debug_print_huffman) {
    //   PrintSubarray("Histogram::freq_subarray", workspace.freq_subarray);
    // }

    // GetCodebook(dict_size, workspace.freq_subarray, workspace.codebook_subarray,
    //             workspace.decodebook_subarray, workspace, queue_idx);
    // if (debug_print_huffman) {
    //   PrintSubarray("GetCodebook::codebook_subarray",
    //                 workspace.codebook_subarray);
    //   PrintSubarray("GetCodebook::decodebook_subarray",
    //                 workspace.decodebook_subarray);
    // }

    // DeviceLauncher<DeviceType>::Execute(
    //     EncodeFixedLenKernel<Q, H, DeviceType>(primary_subarray,
    //                                            workspace.huff_subarray,
    //                                            workspace.codebook_subarray),
    //     queue_idx);

    // if (debug_print_huffman) {
    //   PrintSubarray("EncodeFixedLen::huff_subarray", workspace.huff_subarray);
    // }

    // // deflate
    // DeviceLauncher<DeviceType>::Execute(
    //     DeflateKernel<H, DeviceType>(workspace.huff_subarray,
    //                                  workspace.huff_bitwidths_subarray,
    //                                  chunk_size),
    //     queue_idx);

    // if (debug_print_huffman) {
    //   PrintSubarray("Deflate::huff_subarray", workspace.huff_subarray);
    //   PrintSubarray("Deflate::huff_bitwidths_subarray",
    //                 workspace.huff_bitwidths_subarray);
    // }

    std::vector<Byte> huff_buffer;
    size_t huff_buffer_size = 0;
    if(1){
    huff_buffer.resize(primary_count * sizeof(Q) * 2);
    auto sz3_huffman_encoder = SZ3::HuffmanEncoder<Q>(); 
    sz3_huffman_encoder.preprocess_encode(primary_subarray.data(),primary_count, 0);
    Byte* buffer_pos = huff_buffer.data();
    sz3_huffman_encoder.save(buffer_pos);
    sz3_huffman_encoder.encode(primary_subarray.data(),primary_count, buffer_pos );
    sz3_huffman_encoder.postprocess_encode();
    huff_buffer_size = buffer_pos - huff_buffer.data();
    }
    // write the huffer buffer to the compressed_data 




    // auto nchunk = (primary_count - 1) / chunk_size + 1;
    // size_t *h_meta = new size_t[nchunk * 3]();
    // size_t *dH_uInt_meta = h_meta;
    // size_t *dH_bit_meta = h_meta + nchunk;
    // size_t *dH_uInt_entry = h_meta + nchunk * 2;

    // MemoryManager<DeviceType>().Copy1D(dH_bit_meta,
    //                                    workspace.huff_bitwidths_subarray.data(),
    //                                    nchunk, queue_idx);
    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // // transform in uInt
    // memcpy(dH_uInt_meta, dH_bit_meta, nchunk * sizeof(size_t));
    // std::for_each(dH_uInt_meta, dH_uInt_meta + nchunk,
    //               [&](size_t &i) { i = (i - 1) / (sizeof(H) * 8) + 1; });
    // // make it entries
    // memcpy(dH_uInt_entry + 1, dH_uInt_meta, (nchunk - 1) * sizeof(size_t));
    // for (auto i = 1; i < nchunk; i++)
    //   dH_uInt_entry[i] += dH_uInt_entry[i - 1];

    // // sum bits from each chunk
    // auto total_bits =
    //     std::accumulate(dH_bit_meta, dH_bit_meta + nchunk, (size_t)0);
    // auto total_uInts =
    //     std::accumulate(dH_uInt_meta, dH_uInt_meta + nchunk, (size_t)0);

    // printf("huffman encode time: %.6f s\n", time_span.count());

    // out_meta: |outlier count|outlier idx|outlier data|primary count|dict
    // size|chunk size|huffmeta size|huffmeta|decodebook size|decodebook|
    // out_data: |huffman data|

    // size_t type_bw = sizeof(H) * 8;
    // size_t decodebook_size = workspace.decodebook_subarray.shape(0);
    // size_t huffmeta_size = 2 * nchunk;

    // size_t ddata_size = total_uInts;

    SIZE byte_offset = 0;
    // advance_with_align<size_t>(byte_offset, 1);
    // advance_with_align<int>(byte_offset, 1);
    // advance_with_align<int>(byte_offset, 1);
    // advance_with_align<size_t>(byte_offset, 1);
    // advance_with_align<size_t>(byte_offset, huffmeta_size);
    // advance_with_align<size_t>(byte_offset, 1);
    // advance_with_align<uint8_t>(
    //     byte_offset, (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size));
    // advance_with_align<size_t>(byte_offset, 1);
    // advance_with_align<H>(byte_offset, ddata_size);
    advance_with_align<size_t>(byte_offset, 1); // primary count
    advance_with_align<size_t> (byte_offset, 1); // huff buffer size
    advance_with_align<Byte> (byte_offset, huff_buffer_size); // huff buffer

    // outliter
    advance_with_align<ATOMIC_IDX>(byte_offset, 1);
    advance_with_align<ATOMIC_IDX>(byte_offset, workspace.outlier_count);
    advance_with_align<S>(byte_offset, workspace.outlier_count);
    // advance_with_align<H>(byte_offset, ddata_size);


    compressed_data.resize({(SIZE)(byte_offset)});
    SubArray compressed_data_subarray(compressed_data);

    byte_offset = 0;
    SerializeArray<size_t>(compressed_data_subarray, &primary_count, 1,
                           byte_offset, queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &huff_buffer_size, 1, byte_offset, queue_idx);
    SerializeArray<Byte>(compressed_data_subarray, huff_buffer.data(), huff_buffer_size, byte_offset, queue_idx);
    writefile__("huff_buffer.bin", huff_buffer.data(), huff_buffer_size);

    // inplace huffman decode test 
    {
    auto sz3_huffman_decoder = SZ3::HuffmanEncoder<Q>(); 
    std::vector<Byte> huff_buffer_copy;
    huff_buffer_copy.resize(huff_buffer_size);
    std::copy(huff_buffer.data(), huff_buffer.data() + huff_buffer_size, huff_buffer_copy.data());
    
    huff_buffer_copy.resize(primary_count * sizeof(Q) * 2);
    size_t remaining_length = huff_buffer_size; 
    Byte const * buffer_pos = huff_buffer_copy.data();
    sz3_huffman_decoder.load(buffer_pos, remaining_length);
    auto quantized_data = sz3_huffman_decoder.decode(buffer_pos, primary_count);
    sz3_huffman_decoder.postprocess_decode();
    writefile__("quant_inds_huff.i64",quantized_data.data(), primary_count);
    }

    std::cout << "de_huff_buffer_size: " << huff_buffer_size << std::endl;
    std::cout << "de_byte_offset: " << byte_offset << std::endl;



    // SerializeArray<size_t>(compressed_data_subarray, &primary_count, 1,
    //                        byte_offset, queue_idx);
    // SerializeArray<int>(compressed_data_subarray, &dict_size, 1, byte_offset,
    //                     queue_idx);
    // SerializeArray<int>(compressed_data_subarray, &chunk_size, 1, byte_offset,
    //                     queue_idx);
    // SerializeArray<size_t>(compressed_data_subarray, &huffmeta_size, 1,
    //                        byte_offset, queue_idx);
    // SerializeArray<size_t>(compressed_data_subarray, dH_bit_meta, huffmeta_size,
    //                        byte_offset, queue_idx);
    // SerializeArray<size_t>(compressed_data_subarray, &decodebook_size, 1,
    //                        byte_offset, queue_idx);
    // SerializeArray<uint8_t>(compressed_data_subarray,
    //                         workspace.decodebook_subarray.data(),
    //                         (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size),
    //                         byte_offset, queue_idx);
    // SerializeArray<size_t>(compressed_data_subarray, &ddata_size, 1,
    //                        byte_offset, queue_idx);

    // align_byte_offset<H>(byte_offset);

    // MemoryManager<DeviceType>::Copy1D(
    //     workspace.condense_write_offsets_subarray.data(), dH_uInt_entry, nchunk,
    //     queue_idx);
    // MemoryManager<DeviceType>::Copy1D(
    //     workspace.condense_actual_lengths_subarray.data(), dH_uInt_meta, nchunk,
    //     queue_idx);
    // SubArray<1, H, DeviceType> compressed_data_cast_subarray(
    //     {(SIZE)ddata_size}, (H *)compressed_data_subarray(byte_offset));
    // DeviceLauncher<DeviceType>::Execute(
    //     CondenseKernel<H, DeviceType>(
    //         workspace.huff_subarray, workspace.condense_write_offsets_subarray,
    //         workspace.condense_actual_lengths_subarray,
    //         compressed_data_cast_subarray, chunk_size),
    //     queue_idx);

    // advance_with_align<H>(byte_offset, ddata_size);

    // outlier
    SerializeArray<ATOMIC_IDX>(compressed_data_subarray,
                               &workspace.outlier_count, 1, byte_offset,
                               queue_idx);
    SerializeArray<ATOMIC_IDX>(compressed_data_subarray,
                               workspace.outlier_idx_subarray.data(),
                               workspace.outlier_count, byte_offset, queue_idx);
    SerializeArray<S>(compressed_data_subarray,
                      workspace.outlier_subarray.data(),
                      workspace.outlier_count, byte_offset, queue_idx);
    // use sz3 read and write
    if(0){
      Byte* cmp_ptr = compressed_data.data();
      SZ3::write(primary_count,cmp_ptr ); 
      SZ3::write(huff_buffer_size, cmp_ptr);
      SZ3::write(huff_buffer.data(), huff_buffer_size, cmp_ptr);
      SZ3::write(workspace.outlier_count, cmp_ptr);
      SZ3::write(workspace.outlier_idx_subarray.data(), workspace.outlier_count, cmp_ptr);
      SZ3::write(workspace.outlier_subarray.data(), workspace.outlier_count, cmp_ptr);
    }

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // delete[] h_meta;

    // log::info("Huffman block size: " + std::to_string(chunk_size));
    // log::info("Huffman dictionary size: " + std::to_string(dict_size));
    // log::info("Huffman compress ratio (primary): " +
    //           std::to_string(primary_count * sizeof(Q)) + "/" +
    //           std::to_string(ddata_size * sizeof(H)) + " (" +
    //           std::to_string((double)primary_count * sizeof(Q) /
    //                          (ddata_size * sizeof(H))) +
    //           ")");
    // log::info(
    //     "Huffman compress ratio: " + std::to_string(primary_count * sizeof(Q)) +
    //     "/" + std::to_string(compressed_data.shape(0)) + " (" +
    //     std::to_string((double)primary_count * sizeof(Q) /
    //                    compressed_data.shape(0)) +
    //     ")");

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman compress");
      log::info("Huffman compression throughput: " +
                std::to_string((double)primary_count * sizeof(Q) / timer.get() /
                               1e9) +
                " GB/s");
      timer.clear();
    }
  }

  void DecompressPrimary(Array<1, Byte, DeviceType> &compressed_data,
                         Array<1, Q, DeviceType> &primary_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray compressed_subarray(compressed_data);
    size_t primary_count;
    // int dict_size;
    // int chunk_size;
    // size_t huffmeta_size;
    // size_t *huffmeta;
    // size_t decodebook_size;
    // uint8_t *decodebook;
    // size_t ddata_size;
    ATOMIC_IDX outlier_count;
    ATOMIC_IDX *outlier_idx;
    S *outlier;

    size_t *primary_count_ptr = &primary_count;
    // int *dict_size_ptr = &dict_size;
    // int *chunk_size_ptr = &chunk_size;
    // size_t *huffmeta_size_ptr = &huffmeta_size;
    // size_t *decodebook_size_ptr = &decodebook_size;
    // size_t *ddata_size_ptr = &ddata_size;
    ATOMIC_IDX *outlier_count_ptr = &workspace.outlier_count;

    // H *ddata;

    SIZE byte_offset = 0;

    size_t huff_buffer_size = 0;
    size_t *huff_buffer_size_ptr = &huff_buffer_size;
    DeserializeArray<size_t>(compressed_subarray, primary_count_ptr, 1,
                             byte_offset, false, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, huff_buffer_size_ptr, 1, 
                            byte_offset, false, queue_idx);
    std::vector<Byte> huff_buffer;
    huff_buffer.resize(huff_buffer_size);
    Byte* huff_buffer_ptr = huff_buffer.data();
    std::cout << "de_huff_buffer_size: " << huff_buffer_size << std::endl;
    std::cout << "de_byte_offset: " << byte_offset << std::endl;
    DeserializeArray<Byte>(compressed_subarray, huff_buffer_ptr,
                       huff_buffer_size, byte_offset, false, queue_idx);
    std::cout << "de_byte_offset: " << byte_offset << std::endl;
    writefile__("de_huff_buffer.bin", huff_buffer.data(), huff_buffer_size);
    // DeserializeArray<int>(compressed_subarray, dict_size_ptr, 1, byte_offset,
    //                       false, queue_idx);
    // DeserializeArray<int>(compressed_subarray, chunk_size_ptr, 1, byte_offset,
    //                       false, queue_idx);
    // DeserializeArray<size_t>(compressed_subarray, huffmeta_size_ptr, 1,
    //                          byte_offset, false, queue_idx);
    // DeserializeArray<size_t>(compressed_subarray, huffmeta, huffmeta_size,
    //                          byte_offset, true, queue_idx);
    // DeserializeArray<size_t>(compressed_subarray, decodebook_size_ptr, 1,
    //                          byte_offset, false, queue_idx);
    // DeserializeArray<uint8_t>(compressed_subarray, decodebook, decodebook_size,
    //                           byte_offset, true, queue_idx);
    // DeserializeArray<size_t>(compressed_subarray, ddata_size_ptr, 1,
    //                          byte_offset, false, queue_idx);
    // DeserializeArray<H>(compressed_subarray, ddata, ddata_size, byte_offset,
    //                     true, queue_idx);


    // huff_buffer.resize(primary_count * sizeof(Q) * 2);
    // Byte* huff_buffer_ptr = huff_buffer.data();

    // std::cout << "de_huff_buffer_size: " << huff_buffer_size << std::endl;
    // std::cout << "de_byte_offset: " << byte_offset << std::endl;
    // std::cout<< "writing to de_huff_buffer.bin ...\n";
    // 
    


    // outlier
    DeserializeArray<ATOMIC_IDX>(compressed_subarray, outlier_count_ptr, 1,
                                 byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<ATOMIC_IDX>(compressed_subarray, outlier_idx,
                                 workspace.outlier_count, byte_offset, true,
                                 queue_idx);
    DeserializeArray<S>(compressed_subarray, outlier, workspace.outlier_count,
                        byte_offset, true, queue_idx);
    // use sz3 read
    if(0){
      Byte const * cmp_ptr = compressed_data.data();
      SZ3::read(primary_count, cmp_ptr);
      SZ3::read(huff_buffer_size, cmp_ptr);
      huff_buffer.resize(huff_buffer_size);
      SZ3::read(huff_buffer.data(), huff_buffer_size, cmp_ptr);
      SZ3::read(workspace.outlier_count, cmp_ptr);
      SZ3::read(workspace.outlier_idx_subarray.data(), workspace.outlier_count, cmp_ptr);
      SZ3::read(workspace.outlier_subarray.data(), workspace.outlier_count, cmp_ptr);
      writefile__("de_huff_buffer.bin", huff_buffer.data(), huff_buffer_size);
    }

    workspace.outlier_idx_subarray = SubArray<1, ATOMIC_IDX, DeviceType>(
        {(SIZE)workspace.outlier_count}, outlier_idx);
    workspace.outlier_subarray =
        SubArray<1, S, DeviceType>({(SIZE)workspace.outlier_count}, outlier);

    // SubArray<1, H, DeviceType> ddata_subarray({(SIZE)ddata_size}, ddata);
    // SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size},
    //                                                   huffmeta);
    primary_data.resize({(SIZE)primary_count});
    // decode

    auto sz3_huffman_decoder = SZ3::HuffmanEncoder<Q>(); 
    size_t remaining_length = huff_buffer_size; 
    Byte const * buffer_pos = huff_buffer.data();
    sz3_huffman_decoder.load(buffer_pos, remaining_length);
    auto quantized_data = sz3_huffman_decoder.decode(buffer_pos, primary_count);
    sz3_huffman_decoder.postprocess_decode();
    std::copy(quantized_data.data(), quantized_data.data() + primary_count, primary_data.data());


    // SubArray primary_subarray(primary_data);

    // SubArray<1, uint8_t, DeviceType> decodebook_subarray(
    //     {(SIZE)decodebook_size}, decodebook);
    // int nchunk = (primary_count - 1) / chunk_size + 1;
    // Decode<Q, H, DeviceType>(
    //     ddata_subarray, huffmeta_subarray, primary_subarray, primary_count,
    //     chunk_size, nchunk, decodebook_subarray, decodebook_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman decompress");
      log::info("Huffman decompression throughput: " +
                std::to_string((double)primary_count * sizeof(Q) / timer.get() /
                               1e9) +
                " GB/s");
      timer.clear();
    }
  }

  void Compress(Array<1, S, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    DeviceLauncher<DeviceType>::Execute(
        DictionaryShiftKernel<S, MGARDX_SHIFT_DICT, DeviceType>(
            SubArray(original_data), dict_size),
        queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_SEPARATE_OUTLIER, DeviceType>(
            SubArray(original_data), dict_size,
            workspace.outlier_count_subarray, workspace.outlier_idx_subarray,
            workspace.outlier_subarray),
        queue_idx);
    MemoryManager<DeviceType>::Copy1D(&workspace.outlier_count,
                                      workspace.outlier_count_subarray.data(),
                                      1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (workspace.outlier_count <= workspace.outlier_subarray.shape(0)) {
      // outlier buffer has sufficient size
      log::info("Outlier ratio: " + std::to_string(workspace.outlier_count) +
                "/" + std::to_string(original_data.shape(0)) + " (" +
                std::to_string((double)100 * workspace.outlier_count /
                               original_data.shape(0)) +
                "%)");
    } else {
      log::err("Not enough workspace for outliers.");
      exit(-1);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman outlier");
      timer.clear();
    }

    // Cast to unsigned type
    Array<1, Q, DeviceType> primary_data({original_data.shape(0)},
                                         (Q *)original_data.data());
    CompressPrimary(primary_data, compressed_data, queue_idx);
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, S, DeviceType> &decompressed_data, int queue_idx) {

    // Cast to unsigned type.
    // We use temporarily use size 1 as it we be resized to the correct size.
    Array<1, Q, DeviceType> primary_data({1}, (Q *)decompressed_data.data());

    DecompressPrimary(compressed_data, primary_data, queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_RESTORE_OUTLIER, DeviceType>(
            decompressed_data, dict_size, workspace.outlier_count_subarray,
            workspace.outlier_idx_subarray, workspace.outlier_subarray),
        queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        DictionaryShiftKernel<S, MGARDX_RESTORE_DICT, DeviceType>(
            decompressed_data, dict_size),
        queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman outlier");
      timer.clear();
    }
  }

  bool initialized;
  SIZE max_size;
  int dict_size;
  int chunk_size;
  HuffmanWorkspace<Q, S, H, DeviceType> workspace;
};

} // namespace mgard_x

#endif
