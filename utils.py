import lz4.frame
import pickle


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
