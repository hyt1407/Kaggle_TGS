#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.models as Models
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


# In[2]:


class saveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()    


# In[3]:


class unetUpSampleBlock(nn.Module):
    """
    用于创建unet右侧的上采样层，采用转置卷积进行上采样（尺寸×2）
    self.tranConv将上一层进行上采样，尺寸×2
    self.conv，将左侧特征图再做一次卷积减少通道数，所以尺寸不变
    此时两者尺寸正好一致-----建立在图片尺寸为128×128的基础上，否则上采样不能简单的×2
    """
    def __init__(self,in_channels,feature_channels,out_channels,dp=False,ps=0.25):#注意，out_channels 是最终输出通道的一半。
        super(unetUpSampleBlock,self).__init__()
        self.tranConv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,bias=False)#输出尺寸正好为输入尺寸的两倍
        self.conv = nn.Conv2d(feature_channels,out_channels,1,bias=False) #这一层将传来的特征图再做一次卷积，将特征图通道数减半
        self.bn = nn.BatchNorm2d(out_channels*2) #将特征图与上采样再通道出相加后再一起归一化
        self.dp = dp
        if dp:
            self.dropout = nn.Dropout(ps,inplace=True)
            
    def forward(self,x,features):
        x1 = self.tranConv(x)
        x2 = self.conv(features)
        x = torch.cat([x1,x2],dim=1)
        x = self.bn(F.relu(x))
        return self.dropout(x) if self.dp else x


# In[4]:


class UNet(nn.Module):
    """
    创建unet模型
    """
    def __init__(self,model,drop_i=False,ps_i=None,up_drop = False,ps=None):
        """
        ps:指定dropout的丢失概率
        """
        super(UNet,self).__init__()
        self.down_sample = nn.Sequential(*list(model.children())[:8])#左侧降采样
        self.features = [saveFeatures(list(model.children())[i]) for i in [2,4,5,6]]#获取左侧输出的特征图用于加到右侧输出中
        self.drop_i = drop_i
        if drop_i:
            self.dropout = nn.Dropout(ps_i,inplace=True)
        if ps_i is None:
            ps_i=0.1 #从第二次前向流动开始，pa_i就变成这个了
        if ps is not None:
            assert len(ps)==4  #如果指定了ps，但又没有指定正确的格式，后面的计算就无法进行下去了，此时，抛出一个异常
        else:
            ps = [0.1]*4
        self.up1 = unetUpSampleBlock(512,256,128,up_drop,ps[0])
        self.up2 = unetUpSampleBlock(256,128,64,up_drop,ps[1])
        self.up3 = unetUpSampleBlock(128,64,32,up_drop,ps[2])
        self.up4 = unetUpSampleBlock(64,64,32,up_drop,ps[3])
        self.up5 = nn.ConvTranspose2d(64,1,2,2)
    
    
    def forward(self,x):
        x = F.relu(self.down_sample(x))
        x = self.dropout(x) if self.drop_i else x
        x = self.up1(x,self.features[3].features)
        x = self.up2(x,self.features[2].features)
        x = self.up3(x,self.features[1].features)
        x = self.up4(x,self.features[0].features)
        x = self.up5(x)
        return x[:,0]        #返回(batch_size,1,128,128)，其中1是通道数。如果是x[:,0]就是返回(batch_size,128,128)。
                        #结合下面datasets类可以看出还是return[x:,0]更合适
    
    def close(self):
        for i in self.features:
            i.remove() #记得移除钩子


# In[ ]:




