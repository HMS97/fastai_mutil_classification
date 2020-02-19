import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from path import Path
from torch import nn
from torchvision.transforms.functional import to_tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision import datasets, transforms
from skimage import io
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torch.utils.data import Subset
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.models as models
from sklearn.metrics import accuracy_score



batch_size = 16

transform = transforms.Compose(
    [ transforms.Resize((224,224)),
    transforms.ToTensor(),
     transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    ])

train_dataset = datasets.ImageFolder('./train_dataset/', transform=transform)
val_dataset = datasets.ImageFolder('./test_dataset/', transform=transform)
train_loader=torch.utils.data.DataLoader(train_dataset,
        batch_size=12, shuffle=False)
val_loader=torch.utils.data.DataLoader(val_dataset,
        batch_size=12, shuffle=False)
device = 'cuda'


model = torch.load('alexnet.pkl')
model = torch.load('vgg16.pkl')
model[-1] = model[-1][:-2]


model = model.to(device)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
model.fc = Identity()
# model.classifier = model.classifier[:-3]
embeddings_list = []
result = {}
import numpy as np
loader_dic = {'train':train_loader,'val':val_loader}
dataset_dic = {'train':train_dataset,'val':val_dataset}
for item in ['train','val']:
    loader = loader_dic[item]
    dataset = dataset_dic[item]
    flage = 0
    for images, label in loader:
        images = images.to(device)
        out = model(images).detach().cpu().numpy()
        if flage == 0:
            feature = out
            flage = 1
        else:
            feature = np.concatenate([feature, out],0)

    path_list = []
    for i in range(len(loader.dataset)):
        path_list.append(dataset.imgs[i][0])

    label_list = []
    for i in range(len(loader.dataset)):
        label_list.append(dataset.imgs[i][1])
    result[item]  = {"featrue":feature, "file_name":path_list, "label":label_list}


def most_frequent(List): 
    return max(set(List), key = List.count) 


topk_result = []
for topK in [3,5,10,15,20,30,40,50]:
    confidence_map = cosine_similarity(result['val']['featrue'], result['train']['featrue'])
    sort_map = confidence_map.argsort()
    topK_map = sort_map[:,list(range(len(result['val']['featrue'])-1,len(result['val']['featrue'])-topK-1,-1))]
    

    val_vote = []
    for index in range(len(topK_map)):
        temp_list = topK_map[index]
        label_list = []
        for i in temp_list:
            label_list.append(result['train']['label'][i])
        val_vote.append(most_frequent(label_list))
    y_pred = val_vote 
    y_true = val_dataset.targets
    topk_result.append(round(accuracy_score(y_true, y_pred),3))
    print(f" topK: {topK}  ", round(accuracy_score(y_true, y_pred),3))


model_vote_result = {}
model_vote_result['alexnet'] = []
for i in range(0,1400,200):
    model_vote_result['alexnet'].append(round(accuracy_score(y_true[i:i+199], y_pred[i:i+199]),3))
model_vote_result['alexnet'].append(round(accuracy_score(y_true, y_pred),3))


##draw the accuracy depends on different Topk
# fig, ax = plt.subplots()
# a=ax.get_xticks().tolist()
# a = [3,5,10,15,20,30,40,50]
# ax.set_xticklabels(a)
# ax.plot(resnet50_512,label = 'resnet50')
# ax.plot(alexnet_512,label = 'alexnet')
# ax.plot(vgg16_512,label = 'vgg16')

# ax.legend(loc='upper right')
# fig.show()