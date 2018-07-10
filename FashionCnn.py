# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:42:43 2018

@author: hubinbin
"""
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from fashion_mnist.FashionData import MyDataset
from model.FashionNet import MyFashionNet

root = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(root,"fashion_mnist")

# -----------------ready the dataset--------------------------
train_data=MyDataset(txt=os.path.join(root, 'train.txt'), transform=transforms.ToTensor())
test_data=MyDataset(txt=os.path.join(root, 'test.txt'), transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

#-----------------create the Net and training------------------------
model = MyFashionNet()
#print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    model.train()
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        with torch.no_grad():
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
    
torch.save(model, 'MyFashionMnistCnn.pthx')