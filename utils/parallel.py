from multiprocessing import Process

def run_parallel(fn_a, fn_b):
    p1 = Process(target=fn_a)
    p2 = Process(target=fn_b)
    p1.start()
    p2.start()
    p1.join()
    p2.join()