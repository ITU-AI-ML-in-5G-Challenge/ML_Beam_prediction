# -*- coding: utf-8 -*-

import nni
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_NN(nn.Module) :
    def __init__(self) :
        super(Simple_NN, self).__init__() # Simple_NN, self
        
        self.input_size = 20
        self.output_size = 64
        
        self.loss = nn.BCELoss()

        self.hidden_size1 = 128
        self.hidden_size2 = 1024
        self.hidden_size3 = 256
        
        self.in_0 = nn.Linear(self.input_size, 64)   
        self.act_0 = nn.ReLU()
        self.in_1 = nn.Linear(64, self.hidden_size1)   
        self.act_1 = nn.ReLU()
        self.in_2 = nn.Linear(self.hidden_size1, self.hidden_size2)  
        self.act_2 = nn.ReLU()
        self.in_3 = nn.Linear(self.hidden_size2, self.hidden_size3)  
        self.act_3 = nn.ReLU()
        self.in_4 = nn.Linear(self.hidden_size3, 128)  
        self.act_4 = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(128,self.output_size)

    def forward(self, x) :
        x = self.in_0(x)
        x = self.act_0(x)
        x = self.in_1(x)
        x = self.act_1(x)
        x = self.in_2(x)
        x = self.act_2(x)
        x = self.in_3(x)
        x = self.act_3(x)
        x = self.in_4(x)
        x = self.act_4(x)
        output = self.out(x)
        output = self.sig(output)
        return output
    
    def defineLoss(self): # loss
        self.calc_loss = self.loss #defined_losses(loss)

    def trainNN(self, input_data, target):
        # Pass Through NN
        output = self.forward(input_data)

        # Get Loss
        loss = self.loss(output,target)

        # Backpropagate
        loss.backward()

        return loss.item()

    def testNN(self,input_data,target):
        # Pass Through NN
        output = self.forward(input_data)

        # Get Loss
        loss = self.loss(output,target)

        return loss.item(), output