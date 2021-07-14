import torch
import matplotlib.pyplot as plt

#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:4")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

torch.cuda.set_device(3)

import json
import numpy as np
import os
import random
import re
import pickle
import torch
from tqdm.autonotebook import tqdm

from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
PAD_ID = TOKENIZER._convert_token_to_id("[PAD]")

class Data(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, aspects):

        self.labels=aspects
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask

        self.size=len(input_ids)

    @classmethod
    def getReader(cls,low,up,test=None):

        with open("/DATA/kartik_1901ce16/Review_Exhaustiveness/data_new/aspp.pkl",'rb') as out:
            aspp=pickle.load(out)

        with open('/DATA/kartik_1901ce16/Review_Exhaustiveness/data_new/keys.pkl', 'rb') as out:
            keys=pickle.load(out)

        keys1 = keys[low:up]
        aspp = torch.tensor(aspp, dtype = torch.float)
        labels=aspp[keys1]

        if test is not None:
          print('PEERREAD')
          with open("/DATA/kartik_1901ce16/Review_Exhaustiveness/data/peerread.txt",'r') as out:
              data_s=out.readlines()

        else:
          with open("/DATA/kartik_1901ce16/Review_Exhaustiveness/data_new/r_s.json",'r') as out:
              data_s1=json.load(out)
          data_s = [' '.join(data_s1[a]) for a in keys1]

        assert len(labels) == len(data_s)
        assert torch.equal(labels[1], aspp[keys1[1]])

        print("Total number of Reviews", len(labels))

        pbar = tqdm(data_s)

        embeds=[]

        pbar.set_description("Reading Embeddings")
        batch = TOKENIZER(data_s, padding='max_length', max_length=512, truncation=True)
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']

        return cls(input_ids, token_type_ids, attention_mask, labels)


    def __getitem__(self,idx):

        data=self.input_ids[idx]
        l=self.labels[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]

        return data, token_type_ids, attention_mask, l


    def __len__(self):
        return self.size

    @classmethod
    def pad(cls, batch):

        labels_ = torch.zeros(1,7)
        for sample in batch:
          label_ = sample[3].unsqueeze(0)
          labels_ = torch.cat((labels_, label_), 0)

        labels_ = labels_[1:]

        seqlen_list = [len(sample[0]) for sample in batch]

        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [PAD_ID] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        token_type_ids_list = torch.LongTensor(f(1, maxlen))
        attention_mask_list = torch.LongTensor(f(2, maxlen))

        return input_ids_list, token_type_ids_list, attention_mask_list, labels_

def getLoaders (batch_size):

        print('Reading the training Dataset...')
        print()
        train_dataset = Data.getReader(0,19200) #19200
        print()

        print('Reading the validation Dataset...')
        print()
        valid_dataset = Data.getReader(19200, 23200) #23200

        print('Reading the test Dataset...')
        print()
        test_dataset = Data.getReader(23200, 25248) #25248
        
        trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, collate_fn=Data.pad, num_workers=8)
        validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, collate_fn=Data.pad, num_workers=8)
        testloader = DataLoader(dataset=test_dataset, batch_size = batch_size, collate_fn=Data.pad, num_workers=8)
        
        return trainloader, validloader, testloader

trainloader, validloader, testloader = getLoaders(batch_size=32)

print("Length of TrainLoader:",len(trainloader))
print("Length of ValidLoader:",len(validloader))
print("Length of TestLoader:",len(testloader))

import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """Simple classifier which consists of an encoder(BERT) and an MLP."""

    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 7)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # freeze bert or not
        if True:
            self.freeze_BERT()

    def freeze_BERT(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x, token_type_ids=None, attention_mask=None):
        # Take embedding from [CLS] token

        x = self.encoder(x, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask)["last_hidden_state"][:,0,:]
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.sigmoid(self.linear5(x))
        
        return x

text_model = Classifier()
text_model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(text_model.parameters(), lr=0.0001, weight_decay=1e-5)

text_model.train()
result=[]
EPOCH=60

train_out = []
val_out = []
train_true = []
val_true = []
test_out = []
test_true = []
attn_train = []
attn_val = []

loss_log1 = []
loss_log2 = []
for epoch in range(EPOCH):
    final_train_loss=0.0
    final_val_loss=0.0
    l1 = []
    text_model.train()
    for d in tqdm((trainloader),desc="Train epoch {}/{}".format(epoch + 1, EPOCH)):  
        x1, x2, x3, y_train = d
        x1, x2, x3, y_train = x1.to(device), x2.to(device), x3.to(device), y_train.to(device)
        optimizer.zero_grad()
        out = text_model(x1, x2, x3)
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
        x1_val, x2_val, x3_val, y_val=d
        x1_val, x2_val, x3_val, y_val = x1_val.to(device), x2_val.to(device), x3_val.to(device), y_val.to(device)
        
        out_val = text_model(x1_val, x2_val, x3_val)
        if (epoch+1 == EPOCH):
          val_out.append(torch.transpose(out_val,0,1))
          val_true.append(torch.transpose(y_val,0,1))
        loss = criterion(out_val, y_val)
        l2.append(loss.item())
        final_val_loss+=loss.item()

    loss_log2.append(np.average(l2))
    curr_lr = optimizer.param_groups[0]['lr']

    for i,d in enumerate(testloader):
        x1_test, x2_test, x3_test, y_test = d
        x1_test, x2_test, x3_test, y_test = x1_test.to(device), x2_test.to(device), x3_test.to(device), y_test.to(device)
        out_test = text_model(x1_test, x2_test, x3_test)
        if (epoch+1 == EPOCH):
          test_out.append(torch.transpose(out_test,0,1))
          test_true.append(torch.transpose(y_test,0,1))

    print("Epoch {}, loss: {}, val_loss: {}".format(epoch+1, final_train_loss/len(trainloader), final_val_loss/len(validloader)))
    print()

#torch.save(text_model.state_dict(), "/DATA/kartik_1901ce16/Review_Exhaustiveness/ckpt/model_bert_60.pt")

plt.plot(range(len(loss_log1)), loss_log1)
plt.plot(range(len(loss_log2)), loss_log2)
plt.savefig('/DATA/kartik_1901ce16/Review_Exhaustiveness/graphs_new/loss_bert_60.png')

train_out = torch.cat(train_out, 1)
val_out = torch.cat(val_out, 1)
train_true = torch.cat(train_true, 1)
val_true = torch.cat(val_true, 1)
test_out = torch.cat(test_out, 1)
test_true = torch.cat(test_true, 1)

train_out, val_out, train_true, val_true = train_out.cpu(), val_out.cpu(), train_true.cpu(), val_true.cpu()
test_out, test_true = test_out.cpu(), test_true.cpu()

test_out_ = (test_out, test_true)
test_outs = open('/DATA/kartik_1901ce16/Review_Exhaustiveness/bert_test_out.pkl', 'wb')
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
