import pickle
import time
import pandas as pd
import numpy as np
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
print(pd.DataFrame(y_train[:, 1])[0].value_counts())
with open('./y_val.bin', 'rb') as f:
    y_val = pickle.load(f)
print(y_val.shape)
print(pd.DataFrame(y_val[:, 1])[0].value_counts())
with open('./y_test.bin', 'rb') as f:
    y_test = pickle.load(f)
print(y_test.shape)
print(pd.DataFrame(y_test[:, 1])[0].value_counts())
len_list = [8530, 11367, 15935, 16161, 16305, 16457, 16531, 16645, 16679, 16683, 16700, 16707, 16711, 16746, 16746, 16759, 16766, 16768, 16776, 16779, 16779, 16780, 16786, 16791, 16794, 16799, 16811, 16812, 16813, 16814, 16815, 16816, 16816, 16818, 16819, 16824, 16835, 16839, 16839, 16840, 16860, 16860, 16861, 16869, 16869, 16872, 16873, 16873, 16874, 16887, 16888, 16890, 16895, 16897, 16909, 16911, 16919, 16920, 16926, 16928, 16949, 16954, 16965, 16970, 16984, 16987, 16987, 17004, 17019, 17023, 17035, 17058, 17060, 17090, 17199, 17201, 17221, 17263, 17291, 17318, 17360, 17392, 17406, 17535, 18157, 18244, 18306, 18524, 18587, 18708, 18720, 19027, 20336, 20679, 20786, 21176, 21307, 21331, 21449, 22759, 23163, 23588, 23715, 24286, 24358, 24436, 24564, 24821, 24934, 25637, 26258, 26345, 26350, 26569, 26663, 26748, 26864, 26955, 26992, 27140, 27191, 27268, 27274, 27374, 27402, 27466, 27473, 27480, 27505, 27527, 27532, 27563, 27616, 27694, 27710, 27735, 27742, 27778, 27897, 27936, 28016, 28122, 28206, 28216, 28232, 28494, 28502, 28592, 28594, 28603, 28662, 28878, 29006, 29058, 29098, 29204, 29208, 29259, 29316, 29351, 29424, 29602, 29628, 29748, 29847, 29874, 29948, 30099, 30117, 30155, 30190, 30231, 30485, 30955, 31335, 31382, 31485, 31690, 32631, 33714, 34282, 34361, 34718, 37788]
print(sum(len_list))
print(sum(len_list)/len(len_list))

with open('./ml_all_X_train.bin', 'rb') as f:
    X_train = pickle.load(f)
print(X_train.shape)
with open('./ml_all_X_test.bin', 'rb') as f:
    X_test = pickle.load(f)
print(X_test.shape)

# 데이터 당 길이
# csv_list = glob.glob('/Users/oldman/output_csv/*.csv')
# ordered_csv_list = natsort.natsorted(csv_list)
# filename_list = []
# filelen_list = []
#
# total_len = 0
# for i, path in enumerate(ordered_csv_list):
#
#     data = pd.read_csv(path)
#     data_len = len(data)
#
#     total_len += data_len
#     filelen_list.append(data_len)
#     filename_list.append(os.path.basename(path))
#
#     print(f'{i+1}번째 {path} 데이터 길이: ', data_len)

'''
1번째 /Users/oldman/output_csv/1_A.csv 데이터 길이:  26864
2번째 /Users/oldman/output_csv/1_B.csv 데이터 길이:  16776
3번째 /Users/oldman/output_csv/2_A.csv 데이터 길이:  24934
4번째 /Users/oldman/output_csv/2_B.csv 데이터 길이:  16987
5번째 /Users/oldman/output_csv/3_A.csv 데이터 길이:  27274
6번째 /Users/oldman/output_csv/3_B.csv 데이터 길이:  16161
7번째 /Users/oldman/output_csv/4_A.csv 데이터 길이:  24286
8번째 /Users/oldman/output_csv/4_B.csv 데이터 길이:  16869
9번째 /Users/oldman/output_csv/5_A.csv 데이터 길이:  29424
10번째 /Users/oldman/output_csv/5_B.csv 데이터 길이:  16970
11번째 /Users/oldman/output_csv/6_A.csv 데이터 길이:  27140
12번째 /Users/oldman/output_csv/6_B.csv 데이터 길이:  16895
13번째 /Users/oldman/output_csv/7_A.csv 데이터 길이:  30955
14번째 /Users/oldman/output_csv/7_B.csv 데이터 길이:  17318
15번째 /Users/oldman/output_csv/8_A.csv 데이터 길이:  24564
16번째 /Users/oldman/output_csv/8_B.csv 데이터 길이:  16683
17번째 /Users/oldman/output_csv/9_A.csv 데이터 길이:  27527
18번째 /Users/oldman/output_csv/9_B.csv 데이터 길이:  16780
19번째 /Users/oldman/output_csv/10_A.csv 데이터 길이:  31382
20번째 /Users/oldman/output_csv/10_B.csv 데이터 길이:  16920
21번째 /Users/oldman/output_csv/11_A.csv 데이터 길이:  18306
22번째 /Users/oldman/output_csv/11_B.csv 데이터 길이:  16457
23번째 /Users/oldman/output_csv/12_A.csv 데이터 길이:  27473
24번째 /Users/oldman/output_csv/12_B.csv 데이터 길이:  16835
25번째 /Users/oldman/output_csv/13_A.csv 데이터 길이:  30117
26번째 /Users/oldman/output_csv/13_B.csv 데이터 길이:  18708
27번째 /Users/oldman/output_csv/14_A.csv 데이터 길이:  27694
28번째 /Users/oldman/output_csv/14_B.csv 데이터 길이:  17035
29번째 /Users/oldman/output_csv/15_A.csv 데이터 길이:  27505
30번째 /Users/oldman/output_csv/15_B.csv 데이터 길이:  16860
31번째 /Users/oldman/output_csv/16_A.csv 데이터 길이:  29628
32번째 /Users/oldman/output_csv/16_B.csv 데이터 길이:  16700
33번째 /Users/oldman/output_csv/17_A.csv 데이터 길이:  28494
34번째 /Users/oldman/output_csv/17_B.csv 데이터 길이:  17263
35번째 /Users/oldman/output_csv/18_A.csv 데이터 길이:  26955
36번째 /Users/oldman/output_csv/18_B.csv 데이터 길이:  16965
37번째 /Users/oldman/output_csv/19_A.csv 데이터 길이:  22759
38번째 /Users/oldman/output_csv/19_B.csv 데이터 길이:  16768
39번째 /Users/oldman/output_csv/20_A.csv 데이터 길이:  21449
40번째 /Users/oldman/output_csv/20_B.csv 데이터 길이:  16984
41번째 /Users/oldman/output_csv/21_A.csv 데이터 길이:  11367
42번째 /Users/oldman/output_csv/21_B.csv 데이터 길이:  16816
43번째 /Users/oldman/output_csv/22_A.csv 데이터 길이:  27402
44번째 /Users/oldman/output_csv/22_B.csv 데이터 길이:  16911
45번째 /Users/oldman/output_csv/23_A.csv 데이터 길이:  27936
46번째 /Users/oldman/output_csv/23_B.csv 데이터 길이:  17291
47번째 /Users/oldman/output_csv/24_A.csv 데이터 길이:  31335
48번째 /Users/oldman/output_csv/24_B.csv 데이터 길이:  18244
49번째 /Users/oldman/output_csv/25_A.csv 데이터 길이:  26569
50번째 /Users/oldman/output_csv/25_B.csv 데이터 길이:  16887
51번째 /Users/oldman/output_csv/26_A.csv 데이터 길이:  28603
52번째 /Users/oldman/output_csv/26_B.csv 데이터 길이:  17535
53번째 /Users/oldman/output_csv/27_A.csv 데이터 길이:  28016
54번째 /Users/oldman/output_csv/27_B.csv 데이터 길이:  17060
55번째 /Users/oldman/output_csv/28_A.csv 데이터 길이:  32631
56번째 /Users/oldman/output_csv/28_B.csv 데이터 길이:  16679
57번째 /Users/oldman/output_csv/29_A.csv 데이터 길이:  29098
58번째 /Users/oldman/output_csv/29_B.csv 데이터 길이:  16816
59번째 /Users/oldman/output_csv/30_A.csv 데이터 길이:  27742
60번째 /Users/oldman/output_csv/30_B.csv 데이터 길이:  17392
61번째 /Users/oldman/output_csv/31_A.csv 데이터 길이:  29874
62번째 /Users/oldman/output_csv/31_B.csv 데이터 길이:  18524
63번째 /Users/oldman/output_csv/32_A.csv 데이터 길이:  21307
64번째 /Users/oldman/output_csv/32_B.csv 데이터 길이:  16987
65번째 /Users/oldman/output_csv/33_A.csv 데이터 길이:  27735
66번째 /Users/oldman/output_csv/33_B.csv 데이터 길이:  16711
67번째 /Users/oldman/output_csv/34_A.csv 데이터 길이:  31485
68번째 /Users/oldman/output_csv/34_B.csv 데이터 길이:  16909
69번째 /Users/oldman/output_csv/35_A.csv 데이터 길이:  30231
70번째 /Users/oldman/output_csv/35_B.csv 데이터 길이:  18157
71번째 /Users/oldman/output_csv/36_A.csv 데이터 길이:  37788
72번째 /Users/oldman/output_csv/36_B.csv 데이터 길이:  16786
73번째 /Users/oldman/output_csv/37_A.csv 데이터 길이:  30485
74번째 /Users/oldman/output_csv/37_B.csv 데이터 길이:  19027
75번째 /Users/oldman/output_csv/38_A.csv 데이터 길이:  29208
76번째 /Users/oldman/output_csv/38_B.csv 데이터 길이:  16799
77번째 /Users/oldman/output_csv/39_A.csv 데이터 길이:  34282
78번째 /Users/oldman/output_csv/39_B.csv 데이터 길이:  16824
79번째 /Users/oldman/output_csv/40_A.csv 데이터 길이:  27710
80번째 /Users/oldman/output_csv/40_B.csv 데이터 길이:  17090
81번째 /Users/oldman/output_csv/41_A.csv 데이터 길이:  27563
82번째 /Users/oldman/output_csv/41_B.csv 데이터 길이:  16779
83번째 /Users/oldman/output_csv/42_A.csv 데이터 길이:  29748
84번째 /Users/oldman/output_csv/42_B.csv 데이터 길이:  16839
85번째 /Users/oldman/output_csv/43_A.csv 데이터 길이:  29351
86번째 /Users/oldman/output_csv/43_B.csv 데이터 길이:  16814
87번째 /Users/oldman/output_csv/44_A.csv 데이터 길이:  24821
88번째 /Users/oldman/output_csv/44_B.csv 데이터 길이:  16746
89번째 /Users/oldman/output_csv/45_A.csv 데이터 길이:  24436
90번째 /Users/oldman/output_csv/45_B.csv 데이터 길이:  16872
91번째 /Users/oldman/output_csv/46_A.csv 데이터 길이:  27778
92번째 /Users/oldman/output_csv/46_B.csv 데이터 길이:  17023
93번째 /Users/oldman/output_csv/47_A.csv 데이터 길이:  27616
94번째 /Users/oldman/output_csv/47_B.csv 데이터 길이:  16746
95번째 /Users/oldman/output_csv/48_A.csv 데이터 길이:  27532
96번째 /Users/oldman/output_csv/48_B.csv 데이터 길이:  16874
97번째 /Users/oldman/output_csv/49_A.csv 데이터 길이:  20786
98번째 /Users/oldman/output_csv/49_B.csv 데이터 길이:  17201
99번째 /Users/oldman/output_csv/50_A.csv 데이터 길이:  25637
100번째 /Users/oldman/output_csv/50_B.csv 데이터 길이:  16890
101번째 /Users/oldman/output_csv/51_A.csv 데이터 길이:  28662
102번째 /Users/oldman/output_csv/51_B.csv 데이터 길이:  28878
103번째 /Users/oldman/output_csv/52_A.csv 데이터 길이:  34361
104번째 /Users/oldman/output_csv/52_B.csv 데이터 길이:  16779
105번째 /Users/oldman/output_csv/53_A.csv 데이터 길이:  28592
106번째 /Users/oldman/output_csv/53_B.csv 데이터 길이:  17360
107번째 /Users/oldman/output_csv/54_A.csv 데이터 길이:  30190
108번째 /Users/oldman/output_csv/54_B.csv 데이터 길이:  17406
109번째 /Users/oldman/output_csv/55_A.csv 데이터 길이:  24358
110번째 /Users/oldman/output_csv/55_B.csv 데이터 길이:  16813
111번째 /Users/oldman/output_csv/56_A.csv 데이터 길이:  28216
112번째 /Users/oldman/output_csv/56_B.csv 데이터 길이:  16819
113번째 /Users/oldman/output_csv/57_A.csv 데이터 길이:  28594
114번째 /Users/oldman/output_csv/57_B.csv 데이터 길이:  16860
115번째 /Users/oldman/output_csv/58_A.csv 데이터 길이:  29948
116번째 /Users/oldman/output_csv/58_B.csv 데이터 길이:  16815
117번째 /Users/oldman/output_csv/59_A.csv 데이터 길이:  33714
118번째 /Users/oldman/output_csv/59_B.csv 데이터 길이:  16818
119번째 /Users/oldman/output_csv/60_A.csv 데이터 길이:  8530
120번째 /Users/oldman/output_csv/60_B.csv 데이터 길이:  17004
121번째 /Users/oldman/output_csv/61_A.csv 데이터 길이:  26350
122번째 /Users/oldman/output_csv/61_B.csv 데이터 길이:  16305
123번째 /Users/oldman/output_csv/62_A.csv 데이터 길이:  28206
124번째 /Users/oldman/output_csv/62_B.csv 데이터 길이:  16926
125번째 /Users/oldman/output_csv/63_A.csv 데이터 길이:  27466
126번째 /Users/oldman/output_csv/63_B.csv 데이터 길이:  16897
127번째 /Users/oldman/output_csv/64_A.csv 데이터 길이:  29847
128번째 /Users/oldman/output_csv/64_B.csv 데이터 길이:  23163
129번째 /Users/oldman/output_csv/65_A.csv 데이터 길이:  26748
130번째 /Users/oldman/output_csv/65_B.csv 데이터 길이:  16928
131번째 /Users/oldman/output_csv/66_A.csv 데이터 길이:  29602
132번째 /Users/oldman/output_csv/66_B.csv 데이터 길이:  16766
133번째 /Users/oldman/output_csv/67_A.csv 데이터 길이:  29259
134번째 /Users/oldman/output_csv/67_B.csv 데이터 길이:  17199
135번째 /Users/oldman/output_csv/68_A.csv 데이터 길이:  29006
136번째 /Users/oldman/output_csv/68_B.csv 데이터 길이:  16645
137번째 /Users/oldman/output_csv/69_A.csv 데이터 길이:  23715
138번째 /Users/oldman/output_csv/69_B.csv 데이터 길이:  29204
139번째 /Users/oldman/output_csv/70_A.csv 데이터 길이:  26345
140번째 /Users/oldman/output_csv/70_B.csv 데이터 길이:  16791
141번째 /Users/oldman/output_csv/71_A.csv 데이터 길이:  27268
142번째 /Users/oldman/output_csv/71_B.csv 데이터 길이:  17019
143번째 /Users/oldman/output_csv/72_A.csv 데이터 길이:  18587
144번째 /Users/oldman/output_csv/72_B.csv 데이터 길이:  16531
145번째 /Users/oldman/output_csv/73_A.csv 데이터 길이:  21176
146번째 /Users/oldman/output_csv/73_B.csv 데이터 길이:  16954
147번째 /Users/oldman/output_csv/74_A.csv 데이터 길이:  20679
148번째 /Users/oldman/output_csv/74_B.csv 데이터 길이:  28232
149번째 /Users/oldman/output_csv/75_A.csv 데이터 길이:  20336
150번째 /Users/oldman/output_csv/75_B.csv 데이터 길이:  16919
151번째 /Users/oldman/output_csv/76_A.csv 데이터 길이:  28502
152번째 /Users/oldman/output_csv/76_B.csv 데이터 길이:  16949
153번째 /Users/oldman/output_csv/77_A.csv 데이터 길이:  23588
154번째 /Users/oldman/output_csv/77_B.csv 데이터 길이:  18720
155번째 /Users/oldman/output_csv/78_A.csv 데이터 길이:  30155
156번째 /Users/oldman/output_csv/78_B.csv 데이터 길이:  17058
157번째 /Users/oldman/output_csv/79_A.csv 데이터 길이:  26258
158번째 /Users/oldman/output_csv/79_B.csv 데이터 길이:  16873
159번째 /Users/oldman/output_csv/80_A.csv 데이터 길이:  26663
160번째 /Users/oldman/output_csv/80_B.csv 데이터 길이:  16869
161번째 /Users/oldman/output_csv/81_A.csv 데이터 길이:  29316
162번째 /Users/oldman/output_csv/81_B.csv 데이터 길이:  16794
163번째 /Users/oldman/output_csv/82_A.csv 데이터 길이:  27191
164번째 /Users/oldman/output_csv/82_B.csv 데이터 길이:  16707
165번째 /Users/oldman/output_csv/83_A.csv 데이터 길이:  27480
166번째 /Users/oldman/output_csv/83_B.csv 데이터 길이:  16811
167번째 /Users/oldman/output_csv/84_A.csv 데이터 길이:  28122
168번째 /Users/oldman/output_csv/84_B.csv 데이터 길이:  17221
169번째 /Users/oldman/output_csv/85_A.csv 데이터 길이:  21331
170번째 /Users/oldman/output_csv/85_B.csv 데이터 길이:  16812
171번째 /Users/oldman/output_csv/86_A.csv 데이터 길이:  30099
172번째 /Users/oldman/output_csv/86_B.csv 데이터 길이:  16888
173번째 /Users/oldman/output_csv/87_A.csv 데이터 길이:  29058
174번째 /Users/oldman/output_csv/87_B.csv 데이터 길이:  16840
175번째 /Users/oldman/output_csv/88_A.csv 데이터 길이:  27374
176번째 /Users/oldman/output_csv/88_B.csv 데이터 길이:  16759
177번째 /Users/oldman/output_csv/89_A.csv 데이터 길이:  27897
178번째 /Users/oldman/output_csv/89_B.csv 데이터 길이:  15935
179번째 /Users/oldman/output_csv/90_A.csv 데이터 길이:  31690
180번째 /Users/oldman/output_csv/90_B.csv 데이터 길이:  16839
181번째 /Users/oldman/output_csv/91_A.csv 데이터 길이:  34718
182번째 /Users/oldman/output_csv/91_B.csv 데이터 길이:  16873
183번째 /Users/oldman/output_csv/92_A.csv 데이터 길이:  26992
184번째 /Users/oldman/output_csv/92_B.csv 데이터 길이:  16861
'''
# file_len_dict = dict(zip(filename_list, filelen_list))
#
# with open('file_len_dict.bin', 'wb') as file:
#     pickle.dump(file_len_dict, file)
#
# with open('file_len_dict.bin', 'rb') as f:
#     file_len_dict = pickle.load(f)
#
# for i in file_len_dict:
#     total_len += file_len_dict[i]

# print(max(file_len_dict.values()))
# print(min(file_len_dict.values()))
# print(sorted(file_len_dict.values()))
# print(total_len)

print("time: ", time.time() - start)