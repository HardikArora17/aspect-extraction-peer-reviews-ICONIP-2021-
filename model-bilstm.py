# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:0")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

torch.cuda.set_device(4)

import json
import numpy as np
import os
import random
import re
import pickle
import torch
from tqdm.autonotebook import tqdm

# Load pre-trained model tokenizer (vocabulary)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('stsb-roberta-base')
#model = SentenceTransformer('nli-mpnet-base-v2')

os.environ['TOKENIZERS_PARALLELISM']='False'

from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self,embed,aspects):

        self.labels=aspects
        self.embed=embed

        self.size=embed.size()[0]


    @classmethod
    def getReader(cls,low,up):

        with open("/DATA/kartik_1901ce16/Review_Exhaustiveness/data_new/aspp.pkl",'rb') as out:
            aspp=pickle.load(out)

        with open("/DATA/kartik_1901ce16/Review_Exhaustiveness/data_new/r_s.json",'r') as out:
            data_s1=json.load(out)

        with open('/DATA/kartik_1901ce16/Review_Exhaustiveness/data_new/keys.pkl', 'rb') as out:
            keys=pickle.load(out)

        keys1 = keys[low:up]
        aspp = torch.tensor(aspp, dtype = torch.float)
        labels=aspp[keys1]
        data_s = [data_s1[a] for a in keys1]

        assert len(labels) == len(data_s)
        assert torch.equal(labels[1], aspp[keys1[1]])

        print("Total number of Reviews", len(labels))

        pbar = tqdm(data_s)

        embeds=[]
        index = 0
        for v in pbar:
          pbar.set_description("Reading Embeddings")
          if len(v)<=36:
            v=v+[""]*(36-len(v))
          else:
            v=v[0:36]
          encoded=model.encode(v, show_progress_bar=False)
          embeds.append(encoded)
          index+=1
      
        embeds = torch.tensor(embeds)

        return cls(embeds, labels)


    def __getitem__(self,idx):

        data=self.embed[idx]
        l=self.labels[idx]

        return data,l


    def __len__(self):
        return self.size

def getLoaders (batch_size):

        print('Reading the training Dataset...')
        print()
        train_dataset = Data.getReader(0,19200) #5984(for batch_size=32) 5952(for batch_size=64) 5888(for batch_size=128) 18752(for 25k)
        
        print()

        print('Reading the validation Dataset...')
        print()
        valid_dataset = Data.getReader(19200, 23200) #8000 #24812

        print('Reading the test Dataset...')
        print()
        test_dataset = Data.getReader(23200, 25248) #23200:25248
        
        trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, num_workers=8)
        validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, num_workers=8)
        testloader = DataLoader(dataset=test_dataset, batch_size = batch_size, num_workers=8)
        
        return trainloader, validloader, testloader

trainloader,validloader,testloader = getLoaders(batch_size=32)

print("Length of TrainLoader:",len(trainloader))
print("Length of ValidLoader:",len(validloader))
print("Length of TestLoader:",len(testloader))

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BertModel(nn.Module):
    def __init__(self, in_features, out_features):

        super(BertModel, self).__init__()

        self.in_features = in_features   #768
        self.out_features = out_features    #7

        self.flatten=nn.Flatten()
        self.lstm = nn.LSTM(in_features, 512//2, batch_first=True, bidirectional=True) #bidirectional=True
        self.linear0=nn.Linear(512*36, 512*24)
        self.linear0_=nn.Linear(512*24, 512*13)
        self.linear1=nn.Linear(512*13, 512*7)
        self.linear2=nn.Linear(512*7, 512*2)
        self.linear3=nn.Linear(512*2,512)
        self.linear4=nn.Linear(512,256)
        self.linear5=nn.Linear(256,64)
        self.last_dense = nn.Linear(64, self.out_features)
        self.dropout1=nn.Dropout(p=0.4)
        self.dropout2=nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()

        category = torch.rand(512, out_features,requires_grad=True)  #(512,7)
        nn.init.xavier_normal_(category)
        self.category=category.to(device)

    def forward(self, review):

        s_e=review                  #(32,13,768)

        h0 = torch.zeros(2, s_e.size(0), 512 // 2)
        c0 = torch.zeros(2, s_e.size(0), 512 // 2)
        h0, c0 = h0.to(device), c0.to(device)
        s_e, (hn, cn) = self.lstm(s_e, (h0, c0))    #(32,13,512)

        l = torch.reshape(s_e, (32,36*512))

        l = self.relu(self.linear0(l))
        l = self.dropout1(l)
        l = self.relu(self.linear0_(l))
        l = self.dropout1(l)
        l = self.relu(self.linear1(l))
        l = self.dropout1(l)
        l = self.relu(self.linear2(l))
        l = self.dropout2(l)
        l = self.relu(self.linear3(l))
        l = self.dropout2(l)
        l = self.relu(self.linear4(l))
        l = self.dropout2(l)
        l = self.relu(self.linear5(l))

        model_output = self.sigmoid(self.last_dense(l))
      
        return model_output, s_e

text_model = BertModel(768,7)
text_model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(text_model.parameters(), lr=0.0001, weight_decay=1e-3)

text_model.train()
result=[]
EPOCH=40

train_out = []
val_out = []
train_true = []
val_true = []
test_out = []
test_true = []
attn_train = []
attn_val = []
attn_test = []

loss_log1 = []
loss_log2 = []
for epoch in range(EPOCH):
    final_train_loss=0.0
    final_val_loss=0.0
    l1 = []
    text_model.train()
    for d in tqdm((trainloader),desc="Train epoch {}/{}".format(epoch + 1, EPOCH)):  
        x,y_train = d
        x, y_train = x.to(device), y_train.to(device)
        optimizer.zero_grad()
        out, _ = text_model(x)
        if (epoch+1 == EPOCH):
          train_out.append(torch.transpose(out,0,1))
          train_true.append(torch.transpose(y_train,0,1))
        loss = criterion(out, y_train)
        l1.append(loss.item())
        final_train_loss +=loss.item()
        loss.backward()
        optimizer.step()
    loss_log1.append(np.average(l1))

    text_model.eval()
    l2 = []
    for i,d in enumerate(validloader):
        x_val,y_val=d
        x_val, y_val = x_val.to(device), y_val.to(device)
        
        out_val, _ = text_model(x_val)
        if (epoch+1 == EPOCH):
          val_out.append(torch.transpose(out_val,0,1))
          val_true.append(torch.transpose(y_val,0,1))
        loss = criterion(out_val, y_val)
        l2.append(loss.item())
        final_val_loss+=loss.item()

    loss_log2.append(np.average(l2))
    curr_lr = optimizer.param_groups[0]['lr']

    for i,d in enumerate(testloader):
        x_test, y_test = d
        x_test, y_test = x_test.to(device), y_test.to(device)
        out_test, attn_T = text_model(x_test)
        if (epoch+1 == EPOCH):
          test_out.append(torch.transpose(out_test,0,1))
          test_true.append(torch.transpose(y_test,0,1))
          #attn_test.append(torch.tensor(attn_T))

    print("Epoch {}, loss: {}, val_loss: {}".format(epoch+1, final_train_loss/len(trainloader), final_val_loss/len(validloader)))
    print()


#torch.save(text_model.state_dict(), "/DATA/kartik_1901ce16/Review_Exhaustiveness/ckpt/baseline_bilstm.pt")
#torch.save(optimizer.state_dict(), "/DATA/kartik_1901ce16/Review_Exhaustiveness/ckpt/optim_state_dict.pt")
#torch.save(text_model, "/DATA/kartik_1901ce16/Review_Exhaustiveness/ckpt/model.pt")

plt.plot(range(len(loss_log1)), loss_log1)
plt.plot(range(len(loss_log2)), loss_log2)
#plt.savefig('/DATA/kartik_1901ce16/Review_Exhaustiveness/graphs_new/baseline_bilstm.png')

train_out = torch.cat(train_out, 1)
val_out = torch.cat(val_out, 1)
train_true = torch.cat(train_true, 1)
val_true = torch.cat(val_true, 1)
test_out = torch.cat(test_out, 1)
test_true = torch.cat(test_true, 1)
train_out, val_out, train_true, val_true = train_out.cpu(), val_out.cpu(), train_true.cpu(), val_true.cpu()
test_out, test_true = test_out.cpu(), test_true.cpu()
#attn_test = torch.cat(attn_test, 0)
#attn_test = attn_test.cpu()

'''attn_ = (attn_test)
attn_mat = open('/DATA/kartik_1901ce16/Review_Exhaustiveness/5-712bilstm.pkl', 'wb')
pickle.dump(attn_, attn_mat)
attn_mat.close()'''

test_out_ = (test_out, test_true)
test_outs = open('/DATA/kartik_1901ce16/Review_Exhaustiveness/bilstm_test_out.pkl', 'wb')
pickle.dump(test_out_, test_outs)

def labelwise_metrics(pred, true):

  pred = (pred>0.425)

  batch_size = len(pred)

  pred = torch.tensor(pred, dtype=int)
  true = torch.tensor(true, dtype=int)

  from sklearn.metrics import accuracy_score

  for i in range(batch_size):
    acc=accuracy_score(true[i],pred[i])

    epsilon = 1e-7
    confusion_vector = pred[i]/true[i]

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    precision = true_positives/(true_positives+false_positives+epsilon)
    recall = true_positives/(true_positives+false_negatives+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)

    print("Label: {}, acc: {:.3f}, f1: {:.3f}".format(i+1, acc, f1))

  return 0


print('Training...')
labelwise_metrics(train_out, train_true)
print()
print('Validation...')
labelwise_metrics(val_out, val_true)
print()
print('Test...')
labelwise_metrics(test_out, test_true)