from concurrent.futures import ThreadPoolExecutor

class Utils:
    def __init__(self):
        print('Utils Init')

    def run_parallel(self, func, executors, iterator):
        with ThreadPoolExecutor(max_workers = executors) as executor:
            future = [executor.submit(func, each_arg) for each_arg in iterator]
        for each_result in future:
            each_result.result()