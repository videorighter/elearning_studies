import pickle
import numpy as np

with open('./pre_result_target_bi.bin', 'rb') as f:
    target = pickle.load(f)

target2 = []

for i in target:
    if list(i) == [1, 0]:
        target2.append(0) # 미집중
    else:
        target2.append(1) # 집중

unique, counts = np.unique(target2, return_counts=True)

print(dict(zip(unique, counts)))
# {0: 10141, 1: 16885}