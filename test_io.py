import provider as pv
import sys
import time
import numpy as np

def random_shuffle(x):
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    x = x[perm]
    return x, perm

# Start
start_time = time.time()
print('Start testing')

train_files = pv.getDataFiles('trainval_shape_voxel_data_list.txt')
test_files = pv.getDataFiles('test_shape_voxel_data_list.txt')
print('#train_files = {}'.format(train_files.shape[0]))
print('#test_files = {}'.format(test_files.shape[0]))

num_of_files = train_files.shape[0]
train_files, _ = random_shuffle(train_files)

print('|', end='')
for fn in range(num_of_files):
    current_data, current_label = pv.loadDataFile(train_files[fn], 42)
    print('*', end='')
    sys.stdout.flush()
print('|')
print('total time: ', time.time() - start_time)