# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 21:14:04 2018

@author: hubinbin
"""
import torch
from PIL import Image
from torchvision import transforms
from model.FashionNet import MyFashionNet


class Inferrer(object):
    def __init__(self):
        self.eng_net = torch.load('MyFashionMnistCnn.pthx')
    
    def inferrer(self, image, lang='eng',prep=False):
        transform=transforms.ToTensor()
        img_tensor = transform(image)
        out=self.eng_net(torch.unsqueeze(img_tensor,0))
        pred = torch.max(out, 1)[1]
        return pred
    
if __name__ == '__main__':
    api=Inferrer()
    path='fashion_mnist\\test\\0.jpg'
    img=Image.open(path).convert('RGB')
    pred=api.inferrer(img)
    print(int(pred))