import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py
import math

# Read h5 filename list
def getDataFiles(input_file):
    f = open(input_file, 'r')
    a = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]
        a.append(str(line))
    return np.array(a)

# load h5 file data into memory
def loadDataFile(file_name, num_classes, class_map=None):
    assert(Path(file_name).is_file())
    current_file  = h5py.File(file_name, 'r')
    current_data  = np.array(current_file['data'])
    current_label = np.array(current_file['label'])
    current_file.close()

    # print('Load data:')
    # print(current_data.shape)
    # print(current_label.shape)

    # if class_map != None:
        # Not supported currently

    # unlabeled is last class (num_classes-1 since adding the one after)
    current_label[current_label == 255] = num_classes - 1
    
    # needs to be 1-indexed
    current_label += 1 
    
    return current_data, current_label

def readClassesHist(file_name, num_classes):
    counts = np.ones(num_classes)
    counts[0] = counts[num_classes-1] = 0
    if Path(file_name).is_file():
        print('Use custom classes hists.')
        with open(file_name, 'r') as f:
            index = 0
            for line in f:
                val = line.split(' ')[-1]
                if val[-1] == '\n':
                    val = val[:-1]
                counts[index] = int(val)
                index += 1
        counts = counts / sum(counts) # normalize hist
        for i in range(num_classes):
            if counts[i] > 0:
                counts[i] = 1 / math.log(1.2 + counts[i])
    else:
        print('Use default classes hists.')
    print('Weights:')
    print(counts)
    return counts
