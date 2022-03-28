import lz4.frame
import pickle


def compress(data, compression_level=0):
    serialized_byte_arr = pickle.dumps(data)
    cmp_data = lz4.frame.compress(serialized_byte_arr, compression_level)
    return cmp_data


def decompress(cmp_data):
    byte_array = lz4.frame.decompress(cmp_data)
    data = pickle.loads(byte_array)
    return data