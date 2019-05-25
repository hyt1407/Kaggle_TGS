#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import pandas as pd


# In[2]:


class Decode(object):
    def __init__(self,rle_mask,name,shape=(101,101)):
        self.rle_mask = rle_mask
        self.name = name
        self.shape=shape
    def __call__(self,root):
        for name,rle in zip(self.name,self.rle_mask):
            msk = self.rleDecode(rle)
            cv2.imwrite(root,msk)
    def rleDecode(self,mask_rle):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts = starts -1
        ends = starts + lengths
        img = np.zeros(self.shape[0]*self.shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(self.shape).T


# In[11]:


class Encode(object):
    def __init__(self,img_name,img_list,shape=(101,101)):
        self.img_list=img_list
        self.name = img_name
        self.shape = shape
    def __call__(self,root,save=True):
        submission = pd.DataFrame(columns=('id','rle_mask'))
        for i,(name,img) in enumerate(zip(self.name,self.img_list)):
            rle = self.rleEncode(img)
            name = name.split('.')[0]
            submission.loc[i]={'id':name,'rle_mask':rle}
        if save:
            submission.to_csv(root+"submission.csv",index=False)
        else:
            return submission
            
    def rleEncode(self,img):
        img = cv2.resize(img,self.shape,interpolation=cv2.INTER_CUBIC)
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

