import lz4.frame
import pickle
import logging


def compress(data):
    serialized_byte_arr = pickle.dumps(data)
    cmp_data = lz4.frame.compress(serialized_byte_arr)
    return cmp_data


def decompress(cmp_data):
    byte_array = lz4.frame.decompress(cmp_data)
    data = pickle.loads(byte_array)
    return data


def merge_grad(model_a, model_b):
    list_grad_a = []
    list_grad_b = []
    itr = 0
    for pA, pB in zip(model_a.parameters(), model_b.parameters()):
        list_grad_a.append(pA)
        list_grad_b.append(pB)
        avg = pA.grad + pB.grad
        pA.grad = avg.clone()
        pB.grad = avg.clone()
        itr += 1


logger = logging.getLogger('Debug_logger')


def _set_up_logger(save_url):
    global logger
    new_logger = logging.getLogger('Debug_logger')
    new_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(processName)s - %(threadName)s - %(levelname)s - %(message)s')

    stream_handler.setFormatter(formatter)

    new_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=save_url + '/debug.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(threadName)s - %(levelname)s - %(message)s', '%H:%M:%S')

    file_handler.setFormatter(formatter)
    new_logger.addHandler(file_handler)
    logger = new_logger
    