#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from torchvision.transforms.transforms import Compose,ToTensor,Normalize
from torch.utils import data
import os


# In[2]:


class my_TestDatasets(data.Dataset):
    def __init__(self,image_root,image_name,mean = [119.71304156, 119.71304156, 119.71304156],
                 std = [27.5233033, 27.5233033, 27.5233033]):
        self.root = image_root
        self.file=[]
        self.transform = Compose([
            ToTensor()
        ])
            #Normalize(mean,std)
        for name in image_name:
            img_file = os.path.join(self.root,name)
            self.file.append({'img':img_file,'name':name})
    def __getitem__(self,index):
        img = Image.open(self.file[index]['img']).convert('RGB').resize((128,128),Image.ANTIALIAS)
        img = self.transform(img)
        name = self.file[index]['name']
        return img,name
    def __len__(self):
        return len(self.file)


# In[ ]:




