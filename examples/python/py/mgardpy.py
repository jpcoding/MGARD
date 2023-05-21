import sys
import ctypes
from ctypes.util import find_library
import numpy as np

"""
Python API for SZ2/SZ3
"""

class mgard:

    def __init__(self,mgard_path = None):

        """
        init mgard
        :parmmgard library path
        """

        if mgard_path ==None:
            mgard_path = {
                "darwin": "libmgardpy.dylib",
                "windows": "libmgardpy.dll",
            }.get(sys.platform, "libmgardpy.so")

        self.mgard = ctypes.CDLL(mgard_path)
        
        """
        float * compress_decompress(float * data, int N, size_t* dims, float eb, float s, size_t& compressed_size)
        """
        self.mgard.compress_decompress.argtyp =[
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.mgard.compress_decompress.restype = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        
    def compress_decompress(self, data, eb, s):
        """
        compress_decompress(float * data, int N, size_t* dims, float eb, float s, size_t& compressed_size)
        """
        dims = np.array(data.shape, dtype=np.uint64)
        print(dims)

        compressed_size = ctypes.c_size_t(0)
        decompressed_data = self.mgard.compress_decompress(data, len(dims), dims, eb, s, ctypes.byref(compressed_size))

        ratio =(data.size * 4)/ compressed_size.value 
        decompressed_data = np.asarry(decompressed_data[:np.prod(dims)]).reshape(dims)
        return decompressed_data, ratio 
        

