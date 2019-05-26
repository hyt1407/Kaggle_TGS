#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
#from visdom import Visdom


# In[2]:


class Score(object):
    iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    iou_thresholds = torch.from_numpy(iou_thresholds).float()
    def iou(self,img_ture,img_pred):
        """
        计算两张图片的iou
        """
        img_pred = (img_pred>0).float()#预测值中大于0的预测为正像素，像素值设为1。注意，这只是用于掩码
        i = (img_ture*img_pred).sum()
        u = (img_ture+img_pred).sum()-i
        return i / u if u != 0 else u
    def __call__(self,img_true,img_pred,device):
        """
        可批量计算图片的交并比
        对于我的模型，输出img_pred的形状为(batch_size,H,W)
        但是读取的掩码却是(batch_size,channels(1),H,W)
        所以需要调整维度
        """
        img_pred = torch.squeeze(img_pred)#将掩码维度调整为(batch_size,H,W)
        if img_true.device.type == 'cuda':
            self.iou_thresholds = self.iou_thresholds.to(device)
        num_imgs = len(img_true)
        scores = np.zeros(num_imgs)
        for i in range(num_imgs):
            if img_true[i].sum()==img_pred[i].sum()==0:
                scores[i]=1
            else:
                scores[i] = ((self.iou_thresholds<=self.iou(img_true[i],img_pred[i]))).float().mean()
                #计算每张图片在不同阈值下的得到的平均值
        return scores.mean()


# In[3]:

class Train(object):  
    scores=Score()
    def __init__(self,model,train_loader,value_loader,device,cerition = nn.BCEWithLogitsLoss(),lr=0.001,num_epochs=100):
        self.train_loader = train_loader
        self.value_loader = value_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.model = model
        self.device = device
        self.total_step_one_epoch = len(train_loader)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        self.cerition = cerition
        self.schedulr = optim.lr_scheduler.StepLR(self.optimizer,step_size=20,gamma=0.1)
    def __call__(self):
        loss_t = []
        loss_v = []
        iou_s = []
        #vis = Visdom()
        #vis.line([[0.0,0.0]],[0.0],win="loss",opts = dict(title='loss',legend=['train_loss','test_loss']))
        #vis.line([0.0],[0.0],win="iou",opts=dict(title='iou_score'))
        for epoch in range(self.num_epochs):
            self.schedulr.step()
            loss_train = self.train(epoch)
            loss_test,s = self.value()
            print(epoch,':----',loss_train,'----',loss_test,'----',s)
            loss_t.append(loss_train)
            loss_v.append(loss_test)
            iou_s.append(s)
            #vis.line([s.item()],[epoch],win='iou',update='append')
            #vis.line([[loss_train,loss_test]],[epoch],win='loss',update='append')
        return loss_t,iou_s,loss_v
    def train(self,epoch):
        total_loss,nums = 0,0
        self.model.train()
        for i,data in enumerate(self.train_loader):
            
            img,msk = data
            msk = torch.squeeze(msk)
            num = len(img)
            nums+=num
            img = img.to(self.device)
            msk = msk.to(self.device)
            outputs = self.model(img)
            del img
            loss = self.cerition(outputs,msk)
            total_loss+=(loss.item()*num)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i%10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, self.num_epochs, i+1, self.total_step_one_epoch, loss.item()))    
        return total_loss/nums
    def value(self):
        self.model.eval()
        total_loss,nums,s = 0,0,0
        with torch.no_grad():
            for i, (img,msk) in enumerate(self.value_loader):
                msk = torch.squeeze(msk)
                num = len(img)
                img = img.to(self.device)
                msk = msk.to(self.device)
                outputs = self.model(img)
                del img
                loss = self.cerition(outputs,msk)
                total_loss+=(loss.item()*num)
                s += (self.scores(msk,outputs,self.device)*num)
                nums+=num
            return total_loss/nums,s/nums


# In[ ]:




