from path import Path
import numpy as np
from random import sample 
from shutil import copyfile
import os
import shutil

indices = list(range(400))
split = int(0.5 * len(indices))
print(split)
random_seed = 42
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

name = 'train_dataset'
os.makedirs(f'{name}/')
for item in  [str(i.name) for i in Path('RSSCN7/').listdir()]:
    os.makedirs(f'{name}/{item}')
for item in [i for i in Path('RSSCN7/').listdir()]:
    print(item)
    try:
        for j in [item.files()[i] for i in train_indices]:
            copyfile(j, name+'/' + str(item.name)+'/'+str(j.name))
    except:
        pass

name = 'test_dataset'
os.makedirs(f'{name}/')
for item in  [str(i.name) for i in Path('RSSCN7/').listdir()]:
    os.makedirs(f'{name}/{item}')
for item in [i for i in Path('RSSCN7/').listdir()]:
    print(item)
    try:
        for j in [item.files()[i] for i in val_indices]:
            copyfile(j, name+'/' + str(item.name)+'/'+str(j.name))
    except:
        pass