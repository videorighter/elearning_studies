import pickle
import time
import pandas as pd
import glob
import natsort
import os

start = time.time()
with open('./pre_result_input_bi.bin', 'rb') as f:
    input = pickle.load(f)

with open('./pre_result_target_bi.bin', 'rb') as f:
    target = pickle.load(f)


print(input.shape)
print(target.shape)

with open('./X_train.bin', 'rb') as f:
    X_train = pickle.load(f)
print(X_train.shape)
with open('./X_val.bin', 'rb') as f:
    X_val = pickle.load(f)
print(X_val.shape)
with open('./X_test.bin', 'rb') as f:
    X_test = pickle.load(f)
print(X_test.shape)

with open('./y_train.bin', 'rb') as f:
    y_train = pickle.load(f)
print(y_train.shape)
with open('./y_val.bin', 'rb') as f:
    y_val = pickle.load(f)
print(y_val.shape)
with open('./y_test.bin', 'rb') as f:
    y_test = pickle.load(f)
print(y_test.shape)


csv_list = glob.glob('/Users/oldman/output_csv/*.csv')
ordered_csv_list = natsort.natsorted(csv_list)
filename_list = []
filelen_list = []

total_len = 0
for i, path in enumerate(ordered_csv_list):

    data = pd.read_csv(path)
    data_len = len(data)

    total_len += data_len
    filelen_list.append(data_len)
    filename_list.append(os.path.basename(path))

    print(f'{i+1}번째 {path} 데이터 길이: ', data_len)

file_len_dict = dict(zip(filename_list, filelen_list))

with open('file_len_dict.bin', 'wb') as file:
    pickle.dump(file_len_dict, file)

with open('file_len_dict.bin', 'rb') as f:
    file_len_dict = pickle.load(f)

for i in file_len_dict:
    total_len += file_len_dict[i]

print(max(file_len_dict.values()))
print(min(file_len_dict.values()))
print(sorted(file_len_dict.values()))
print(total_len)

print("time: ", time.time() - start)