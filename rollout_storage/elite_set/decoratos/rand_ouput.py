import functools
import random


def rand_output(chance=0, seed=0):
    def rand_output_dec(func):
        @functools.wraps(func)
        def wrapper_dec(self, *args, **kwargs):
            if local_random.random() < chance:
                return local_random.randint(0, self.buf_size)
            return func(self, *args, **kwargs)
        return wrapper_dec
    local_random = random.Random(seed)
    return rand_output_dec
