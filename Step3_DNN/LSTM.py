"""_LSTM class based on torch_
:param param1: a dict content the parameters of MLP architecture and the training parameter

Param={
    'inputsize':M,
    'outputsize':N,
    'LSTM_hidden':L,
    'num_layer':n,
    'learning_rate':0.001,
    'opt':'Adam',
}

:returns: MLP model
:raises keyError: raises an exception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self,param):
        super(Net,self).__init__()
        self.param = param
        self._build_net()
    
    def _build_net(self):
        '''
        self.model = nn.Sequential()
        self.model.add_module('LSTM',
                                nn.LSTM(input_size=self.param['inputsize'],
                                hidden_size=self.param['LSTM_hidden'],
                                num_layers=self.param['num_layer'],
                                batch_first = True,
                                bias=True,dropout=0.5))
        self.model.add_module('output',nn.Linear(self.param['LSTM_hidden'],self.param['outputsize']))
        '''
        
        self.lstm = nn.LSTM(input_size=self.param['inputsize'],
                                hidden_size=self.param['LSTM_hidden'],
                                num_layers=self.param['num_layer'],
                                batch_first = True,
                                bias=True,dropout=0.5)
        self.fc1 = nn.Linear(self.param['LSTM_hidden']*self.param['windowsize'],self.param['windowsize'])
        self.fc2 = nn.Linear(self.param['windowsize'],self.param['windowsize'])
        self.fc3 = nn.Linear(self.param['windowsize'],self.param['outputsize'])
    
    def forward(self,input):
        x, _ = self.lstm(input) #x.shape (batch, seq, feature)        
        b, s, h = x.shape  #x.shape (batch, seq, hidden)
        x = x.reshape(b, s*h)
        x = nn.Tanh()(self.fc1(x))
        x = nn.Tanh()(self.fc2(x))
        x = self.fc3(x)
        out = x.squeeze(1)
        return out
    
if __name__=='__main__':
    x=torch.tensor(np.zeros((10,3,1)),dtype=torch.float32)
    y=torch.tensor(np.zeros((10,3,1)),dtype=torch.float32)
    x , y =(Variable(x),Variable(y))
    
    param={
        'inputsize':1,
        'outputsize':1,
        'LSTM_hidden':20,
        'num_layer':3,
        'learning_rate':0.001,
        'opt':'Adam',
    }
    
    net = Net(param)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
    loss_func = torch.nn.MSELoss()
    for t in range(5000):
        prediction = net(x)
        loss = loss_func(prediction[0],y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        