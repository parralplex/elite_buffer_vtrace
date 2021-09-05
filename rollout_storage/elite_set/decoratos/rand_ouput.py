import functools
import random


def rand_output(chance=0):
    def rand_output_dec(func):
        @functools.wraps(func)
        def wrapper_dec(self, *args, **kwargs):
            if random.random() < chance:
                return random.randint(0, self.buf_size)
            return func(self, *args, **kwargs)
        return wrapper_dec
    return rand_output_dec
