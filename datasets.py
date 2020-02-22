import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class HumanRecognitionDataset(Dataset):

    def __init__(self,train, root_dir="data", train_list= "train.txt",test_list="test.txt",transform=None):
        self.train_list = list()
        self.test_list = list()
        f = open(train_list,"r")
        g = open(test_list, "r")
        for l in f.readlines():
            splitted = l.rstrip().split(" ")
            img = splitted[0]
            if(len(splitted) == 3):
                cls = l.rstrip().split(" ")[2]
            else:
                assert len(splitted) == 2
                cls = l.rstrip().split(" ")[1]
            #print(l.rstrip().split(" ")[2])
            #assert False
            #print(img, cls, " hi")
            if(int(cls)==-1):
                cls = str(0)
            elif(int(cls) ==1):
                cls = str(1)
            else:
                 continue
            self.train_list.append((root_dir + "/" + img+".jpg", cls))
                 
        for l in g.readlines():
            splitted = l.rstrip().split(" ")
            img = splitted[0]
            if(len(splitted) == 3):
                cls = l.rstrip().split(" ")[2]
            else:
                assert len(splitted) == 2
                cls = l.rstrip().split(" ")[1]
           # print(img, cls)
            if(int(cls)==-1):
                cls = str(0)
            elif(int(cls) ==1):
                cls = str(1)
            else:
                 continue
            self.test_list.append((root_dir + "/" + img+".jpg", cls))
        self.transform = transform
        self.train = train
        f.close()
        g.close()

    def __len__(self):
        if(self.train):
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, idx):
        if(self.train):
            cur = self.train_list[idx]
        else:
            cur = self.test_list[idx]
        img = cur[0]
        cls = cur[1]
        image = Image.open(img)
        transformed = self.transform(image)
        print(img, cls)
        return transformed, int(cls)

        
