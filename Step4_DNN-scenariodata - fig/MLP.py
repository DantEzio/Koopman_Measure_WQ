"""_MLP class based on torch_
:param param1: a dict content the parameters of MLP architecture and the training parameter

Param={
    'inputsize':M,
    'outputsize':N,
    'layers':[m1,m2],
    'act':['relu','tanh',],
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

class MLP(nn.Module):
    def __init__(self,param):
        super(MLP,self).__init__()
        self.param = param
        self._build_net()
    
    def _build_net(self):
        self.model = nn.Sequential()
        self.model.add_module('input',nn.Linear(self.param['inputsize'],self.param['layers'][0]))
        self._get_act('input_act')
        for l in range(len(self.param['layers'])-1):
            self.model.add_module('h'+str(l+1),nn.Linear(self.param['layers'][l],self.param['layers'][l+1]))
            self._get_act('h'+str(l+1)+'_act')
        self.model.add_module('output',nn.Linear(self.param['layers'][-1],self.param['outputsize']))
        self._get_act('output_act')
        
    def _get_act(self,name):
        if self.param['act']=='relu':
            self.model.add_module(name,nn.ReLU())
        elif self.param['act']=='tanh':
            self.model.add_module(name,nn.Tanh())
        elif self.param['act']=='sigmoid':
            self.model.add_module(name,nn.Sigmoid())
        elif self.param['act']=='softmax':
            self.model.add_module(name,nn.Softmax())
        else:
            pass
    
    def forward(self,input):
        out = self.model(input)
        return out
    
    
if __name__=='__main__':
    x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y = x.pow(3)+0.1*torch.randn(x.size())
    x , y =(Variable(x),Variable(y))
    
    param={
        'inputsize':1,
        'outputsize':1,
        'layers':[20,30,10,20],
        'act':['relu','tanh','tanh','sigmoid','softmax','tanh'],
        'learning_rate':0.001,
        'opt':'Adam',
    }
    
    net = MLP(param)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
    loss_func = torch.nn.MSELoss()
    for t in range(5000):
        prediction = net(x)
        loss = loss_func(prediction,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t%1000 ==0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.05)