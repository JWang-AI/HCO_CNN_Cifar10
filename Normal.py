import cifar10_multi_gpu_train as ct
import cifar10_eval as ce
import numpy as np
from HCO import bndry_proce
import os
ld = np.array([50, 50, 50, 2, 2, 2, 2, 2, 2, 0, 0, 0])  # low boundary
ud = np.array([180, 180, 180, 7, 7, 7, 7, 7, 7, 1, 1, 1])  # up  boundary
dim = 12
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
record = np.array([], dtype=float)
for i in range(1, 601):
    idv = np.random.rand(dim)
    idv = np.add(ld, np.multiply(idv, np.subtract(ud, ld)))
    idv = bndry_proce(idv)
    ct.main(idv)
    record = np.append(record, ce.main(idv))
    print("this is the %dth epoch " % i)
record = np.append(record, np.mean(record))
record = np.append(record, np.max(record))
fo = open("/Users/wangjun/Documents/study/Graduation_Project/HCOCNN/result_normal.txt", "w")
fo.write(str(record))
fo.close()
