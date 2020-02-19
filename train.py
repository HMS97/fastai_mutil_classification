from fastai.vision import *
from fastai.metrics import accuracy
from path import Path
import numpy as np

size = 224
bs = 32

data = (ImageList.from_folder('./')
        .split_by_folder(train = 'train_dataset', valid = 'test_dataset')
        .label_from_folder()
        .transform(get_transforms(do_flip = False,flip_vert = False), size = size)
#         .add_test_folder('/onepanel/test')
        .databunch(bs = bs)
        .normalize(imagenet_stats))

model_list = ['vgg16','alexnet','resnet50']
for index,i in enumerate([models.vgg16_bn, models.alexnet,models.resnet50,]):
    learner = cnn_learner(data, i, metrics=[accuracy], callback_fns=ShowGraph)
    # learner.load('vgg16.pkl')
    learner.fit_one_cycle(10, max_lr=slice(1e-3, 1e-2))
    modelname = learner.model
    modelname.cpu()
    torch.save(modelname, f'{model_list[index]}.pkl')
