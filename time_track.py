import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Done. ({time.time() - start}s)')
        return result
    return wrapper