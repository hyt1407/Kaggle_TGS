#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
from PIL import Image
import numpy as np
from torchvision.transforms.transforms import Compose,RandomHorizontalFlip,RandomCrop,ToTensor,Normalize
import torchvision.transforms.functional as F
from torch.utils import data
import os


# In[46]:


#在pytorch中提供的基本的transforms包不能同时对image及其mask做增强，故自己重构
#只重构基本函数操作，其余特性继承至原transforms包
class my_Compose(Compose):
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,img,msk):
        for t in self.transforms:
            img,msk = t(img,msk)
        return img,msk


# In[47]:


class my_RandomCrop(RandomCrop):
    def __init__(self,crop_size=100,p = 0.5):
        self.crop_size = crop_size
        self.p = p
    def __call__(self,img,msk):
        if np.random.random()<=self.p:
            image = np.asarray(img)
            mask = np.asarray(msk)
            coordinate = np.random.randint(0,image.shape[0]-self.crop_size)
            img = cv2.copyMakeBorder(image[coordinate:coordinate+self.crop_size,coordinate:coordinate+self.crop_size,:],14,14,14,14,cv2.BORDER_REPLICATE)
            msk = cv2.copyMakeBorder(mask[coordinate:coordinate+self.crop_size,coordinate:coordinate+self.crop_size],14,14,14,14,cv2.BORDER_REPLICATE)
            return img,msk
        return img,msk


# In[48]:


class my_RandomFlip(RandomHorizontalFlip):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img,msk):
        if np.random.random()<=self.p:
            image = np.asarray(img)
            mask = np.asarray(msk)
            flip_code = np.random.choice((-1,0,1))
            return  cv2.flip(image,flip_code,dst=None),cv2.flip(mask,flip_code,dst=None)
        return img,msk


# In[49]:


class my_ToTensor(ToTensor):
    def __call__(self,pic,msk):
        return F.to_tensor(pic),F.to_tensor(msk)


# In[50]:


class my_RandomSwap(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img,msk):
        if np.random.random()<=self.p:
            image = np.asarray(img)
            perms = ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
            swap = perms[np.random.randint(0, len(perms))]
            image = image[:,:,swap]
            return image,msk
        return img,msk


# In[51]:


class my_Histogram(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img,msk):
        if np.random.random()<=self.p:
            image = np.asarray(img)
            mask = np.asarray(msk)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(image)
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            return image,mask
        return img,msk


# In[52]:


class my_Normalize(Normalize):
    def __init(self,mean,std,inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self,img,msk):
        return F.normalize(img, self.mean, self.std, self.inplace),msk


# In[53]:


class my_TrainDataset(data.Dataset):
    def __init__(self,image_root,mask_root,image_name,mean=[120.34612148,120.34612148,120.34612148],
                 std=[27.75959516, 27.75959516, 27.75959516],data_type='train'):
        self.image_root = image_root
        self.mask_root = mask_root
        #self.image_name = image_name
        self.mean = mean
        self.std = std
        self.data_type = data_type
        if self.data_type == 'train':
            self.image = image_name[:int(0.7*len(image_name))]
            self.transform = my_Compose([
                my_Histogram(),
                my_RandomCrop(),
                my_RandomFlip(),
                my_RandomSwap(),
                my_ToTensor()
            ])
            #my_Normalize(self.mean,self.std)
        elif self.data_type == 'value':
            self.image = image_name[int(0.7*len(image_name)):]
            self.transform = my_Compose([
                my_ToTensor()
            ])
            #my_Normalize(self.mean,self.std)
        self.file = []
        for name in self.image:
            img_file = os.path.join(self.image_root,name)
            msk_file = os.path.join(self.mask_root,name)
            self.file.append({
                "img":img_file,
                "msk":msk_file
            })
    
    def __getitem__(self,index):
        datafiles = self.file[index]
        img = Image.open(datafiles["img"]).convert('RGB').resize((128,128),Image.ANTIALIAS)
        msk = Image.open(datafiles["msk"]).convert('L').resize((128,128),Image.ANTIALIAS)
        #print('msk:')
        #print(np.asarray(msk))
        img,msk = self.transform(img,msk)
        return img,msk
    def __len__(self):
        return len(self.image)

