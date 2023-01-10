import os
import math
from multiprocessing import Process


def exe(start, chunk, mode):
    cmd = 'python make_dataset.py --start {} --chunk {}  --mode {}'.format(
        start,
        chunk,
        mode)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    mode = 'train'
    pids = []
    RGB_dir = 'RealBlurSourceFiles'
    num = len(os.listdir(RGB_dir))
    cores = os.cpu_count()
    chunk = math.ceil(num / cores)
    for i in range(cores):
        p = Process(target=exe, args=(i * chunk, chunk, mode))
        p.start()
        pids.append(p.pid)
        os.system("taskset -p -c {} {}".format(i % cores, p.pid))
    print(pids)
