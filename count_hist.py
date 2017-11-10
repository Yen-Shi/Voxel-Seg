import sys
import time
import h5py
import math
import numpy as np
import collections as cl
from pathlib import Path
import provider as pv

# load h5 file data into memory
def loadDataFile(file_name, num_classes, class_map=None):
    assert(Path(file_name).is_file())
    current_file  = h5py.File(file_name, 'r')
    current_label = current_file['label'][()]
    current_file.close()

    # unlabeled is last class (num_classes-1 since adding the one after)
    current_label[current_label == 255] = num_classes - 1
    
    return current_label

# Start
start_time = time.time()
print('Start counting')

train_files = pv.getDataFiles('trainval_shape_voxel_data_list.txt')
test_files = pv.getDataFiles('test_shape_voxel_data_list.txt')
print('#train_files = {}'.format(train_files.shape[0]))
print('#test_files = {}'.format(test_files.shape[0]))

num_of_files = train_files.shape[0]

classes_counter = cl.Counter()
print('|', end='')
for fn in range(num_of_files):
    current_label = loadDataFile(train_files[fn], 42)
    classes_counter = classes_counter + cl.Counter(current_label.ravel())
    print('*', end='')
    sys.stdout.flush()
print('|')
print('total time: ', time.time() - start_time)
example = pv.readClassesHist('classes_hist.txt', 42)
print('Original counter:')
print(classes_counter)

# print('Write classes_hists to ' + 'classes_hist.txt')
# with open('classes_hist.txt', 'w') as f:
#     for i in range(42):
#         if example[i] != 0:
#             f.write('{}\t{}\n'.format(i+1, classes_counter[i]))
#         else:
#             f.write('{}\t{}\n'.format(i+1, 0))

classes_hist = np.zeros(42)
for i in range(42):
    if example[i] != 0:
        classes_hist[i] = classes_counter[i]
classes_hist = classes_hist / sum(classes_hist)

for i in range(42):
    if example[i] != 0:
        if classes_hist[i] != 0:
            example[i] = 1 / math.log(1.2 + classes_hist[i])
        else:
            example[i] = 0
print('Get classes_hists:')
print(example)