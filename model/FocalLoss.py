#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch


# In[2]:
class focalLoss(nn.Module):
    def __init__(self,device,gamma=2,average=True):
        super(focalLoss,self).__init__()
        self.gamma = gamma
        self.average = True
        self.device = device
    def forward(self,pred,target,class_weight=None):
        if class_weight is None:
            class_weight = [1,1]#针对不同类别可以采用不同的权重，解决类别不平衡。具体操作见后
        target = target.view(-1,1).long()#转成long类是为了作为索引下标。
        pred = torch.sigmoid(pred).view(-1,1)
        pred = torch.cat((1-pred,pred),dim=1) #将预测值转为两列，一列为f(x),一列为1-f(x)。方便后面的计算
        #print(target)
        #print(pred)
        select = torch.zeros(len(pred), 2).to(self.device)
        #print(select.shape)
        #print(target.shape)
        select = select.scatter(1,target,1.)#标记每个像素的类别。每行第一个为0类，第二个为1类
        #pred*select得到的结果中，对应类别的取值留下
        pred = (pred*select).sum(1).view(-1,1)
        pred = torch.clamp(pred,1e-8,1-1e-8)
        class_weight = torch.FloatTensor(class_weight).view(-1,1).to(self.device)
        class_weight = torch.gather(class_weight,0,target)#这样对应位置的权重变成了该有的值
        batch_loss = -class_weight*(torch.pow((1-pred),self.gamma))*pred.log()
        return batch_loss.mean()

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss


# In[ ]:




