import sys
import ctypes
import numpy as np
from ctypes.util import find_library

class mgard:

    def __init__(self,mgard_path = None):

        """
        init mgard
        :parmmgard library path
        """

        if mgard_path ==None:
            mgard_path = {
                "darwin": "../install/lib/libmgardpy.dylib",
                "windows": "../install/lib/libmgardpy.dll",
            }.get(sys.platform, "../install/lib/libmgardpy.so")

        self.mgard = ctypes.CDLL(mgard_path)
        
        """
        float * compress_decompress(float * data, int N, size_t* dims, float eb, float s, size_t& compressed_size)
        """
        self.mgard.compress_decompress_float.argtypes =[ctypes.POINTER(ctypes.c_float), 
                                                  ctypes.POINTER(ctypes.c_float), 
                                                  ctypes.c_int, 
                                                  ctypes.POINTER(ctypes.c_size_t),
                                                  ctypes.c_float, 
                                                  ctypes.c_float,
                                                  ctypes.POINTER(ctypes.c_size_t)]
        
    def compress_decompress(self, data, eb, s):
        """
        compress_decompress(float * data, float* ddata, int N, size_t* dims, float eb, float s, size_t& compressed_size)
        """
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # ddata = np.z(data.size, dtype=np.float32)
        ddata = np.zeros(data.shape, dtype=np.float32)
        
        ddata_ptr = ddata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dims = np.asarray(data.shape).astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        N = ctypes.c_int(len(data.shape))
        print(N)
        compressed_size = ctypes.c_size_t()
        # eb = ctypes.c_float(eb)
        # s = ctypes.c_float(s)
        self.mgard.compress_decompress_float(data_ptr, ddata_ptr, N, dims, eb, s, ctypes.byref(compressed_size))

        ratio = (data.size * data.itemsize)/ float(compressed_size.value)
        print(ratio)
        return ddata, ratio
    
    def verify(self, src_data, dec_data):
        """
        Compare the decompressed data with original data
        :param src_data: original data, numpy array
        :param dec_data: decompressed data, numpy array
        :return: max_diff, psnr, nrmse
        """
        data_range = np.max(src_data) - np.min(src_data)
        diff = src_data - dec_data
        max_diff = np.max(abs(diff))
        print("abs err={:.8G}".format(max_diff))
        mse = np.mean(diff ** 2)
        nrmse = np.sqrt(mse) / data_range
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
        return max_diff, psnr, nrmse
        