import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
#import itertools

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

#paths - for linus ssh
project_dir = os.path.dirname(os.path.abspath(__file__))
week1_path = os.path.join(project_dir, 'week1_data_ns.pkl')
to_idx_path = os.path.join(project_dir, 'to_idx.pkl')

week1 = pd.read_pickle(week1_path)
loss2idx = pd.read_pickle(to_idx_path)


#Dataset
class LossDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        #If GPU RAM allows, we can modify the dataframe and send all data to device (GPU) here to speed up the loading process,
        #which may be the bottleneck during training.
    def __getitem__(self, idx):
        center = self.df.iloc[idx]['center']
        contexts = self.df.iloc[idx]['contexts']
        pos_end = self.df.iloc[idx]['pos_end']
        negs = self.df.iloc[idx]['negs']
        neg_end = self.df.iloc[idx]['negs_end']
        return center, contexts, negs, pos_end, neg_end
    def __len__(self):
        return len(self.df)

#Iterable dataset creates generate and is applicable when the dataset is too large to read the index for fetching data.
#Iterable Dataset
# class LossIterableDataset(IterableDataset):
#     def __init__(self, path):
#         self.path = path
#     def parse_file(self, path):
#         with open(path) as f:
#             for line in f:
#                 tokens = line.strip('\n').split('[]')
#     def get_stream(self, path):
#         return itertools.cycle(self.parse_file(path))
#     def __iter__(self):
#         return self.get_stream(self.path)


#self-defined collate_fn for controlling the output of the dataloader; for padding purposes; padding could also be
#completed in the dataset class as noticed
def batchify(data):
    max_pos = max(len(pos) for center, pos, neg, pend, nend in data)
    max_neg = max(len(neg) for center, pos, neg, pend, nend in data)
    centers, poses, negs, pends, nends = [], [], [], [], []
    for center, pos, neg, pend, nend in data:
        cur_pos = len(pos)
        cur_neg = len(neg)
        pad_pos = pos + [0]*(max_pos - cur_pos)
        pad_neg = neg + [0]*(max_neg - cur_neg)
        centers.append(center)
        poses.append(pad_pos)
        negs.append(pad_neg)
        pends.append(pend)
        nends.append(nend)
    return centers, poses, negs, pends, nends

#skipgram with negative sampling in place of softmax. Negative Sampling and Hierarchical Softmax helps speed up computation
#Negative sampling turns the softmax task into multiple sigmoid tasks, which is more computational-effective
class SGNS(nn.module):
    def __init__(self, vocab_size, embed_size):
        super(SGNS, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, input, pos, negs, mask_pos, mask_neg):
        in_vec = self.in_embed(input).unsqueeze(dim=2)  #embedding vector for the input
        pos_vec = self.out_embed(pos)                   #embedding vector for positive outputs
        negs_vec = self.out_embed(negs)                 #embedding vector for nnegative outputs

        #得到vector embedding之后， 相乘得到raw score, sigmoid转化为概率，log转乘法为加法有利于小数计算
        #在计算loss时，vector multiplication得到的是positive pair等于1的概率和negative pair等于0的概率，因此后者加上负号，
        #以此可以同时maximize两者之和
        pos_dot = F.logsigmoid(torch.bmm(pos_vec, in_vec)).squeeze(dim=2)
        neg_dot = F.logsigmoid(torch.bmm(-negs_vec, in_vec)).squeeze(dim=2)
        pos_masked = pos_dot * mask_pos
        neg_masked = neg_dot * mask_neg
        pos_loss = pos_masked.sum(dim=1)
        neg_loss = neg_masked.sum(dim=1)
        #我们需要minimize loss。前面算出的是需要maximize的raw score，因此我们需要加负号成为可以minimize的score
        loss = -(pos_loss + neg_loss).mean()  #一个batch中算出的所有loss取平均值
        return loss


#training
def train_one_epoch(model, dataloader, optimizer, device):
    for targets, positive, negs, pos_end, neg_end in tqdm(dataloader):  
        #By default, dataloader will wrap the output batches into tensors. As we used self-defined batchify collate_fn here,
        #we need to wrap them into tensors here. Alternatively, we can modify the output of batchify function to return tensors.
        #turn the float tensors here to long tensors because the nn.Embedding() layer only accepts long tensors. Float tensors are
        #allowed in nn.Linear() layers.
        #Then, we need to send these tensors to device (GPU / CPU)
        negs = torch.tensor(negs).long().to(device)
        targets = torch.tensor(targets).long().to(device)
        positive = torch.tensor(positive).long().to(device)
        pos_end = torch.tensor(pos_end)
        neg_end = torch.tensor(neg_end)
        #creating masks for the padded positions
        mask_pos = (torch.arange(pos_end.max().item())[None, :] < pos_end[:, None]).float().to(device)
        mask_neg = (torch.arange(neg_end.max().item())[None, :] < neg_end[:, None]).float().to(device)

        loss = model(targets, positive, negs, mask_pos, mask_neg) #compute loss
        optimizer.zero_grad() #zero out previous gradients
        loss.backward() #backpropagation
        optimizer.step() #update weights

    print(f'loss: {loss.item()}')

def train(model, dataloader, optimizer, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_one_epoch(model, dataloader, optimizer, device)
        print('--------------------------')
    print('Training Finished.')


EmbedSize = 20
VocabSize = 16332
LearningRate = 0.002
sgns = SGNS(VocabSize, EmbedSize).to(device)  #send to GPU / CPU
optimizer = torch.optim.Adam(sgns.parameter(), LearningRate)
training_data = LossDataset(week1)
#num_workers matches the number of cores of CPU
#pin_memory helps make the process of sending data from CPU to GPU faster. If we already sent everything to GPU in Dataset class, make sure to set pin_memory=False
#collate_fn allows manipulations over the output of the dataloader, such as padding
training_loader = DataLoader(training_data, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True, collate_fn=batchify)
train(sgns, training_loader, optimizer, device, 100)

# #saving
# torch.save(sgns.state_dict(), os.path.join(project_dir, 'XXX'))
# torch.save(sgns, os.path.join(project_dir, 'XXXX'))
# torch.save(sgns.in_embed, os.path.joinn(project_dir, 'XXXXX'))




