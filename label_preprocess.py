# 2021-01-18 videorighter
# labeling code

import csv

def labeling(path):
    # create label dictionary
    label_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            line = [x for x in line if x]
            label_dict[line[0]] = line[1:]
    f.close()

    return label_dict

if __name__ == '__main__':
    label_dict = labeling('/home/oldman/study/label_100.csv')
    print(label_dict)